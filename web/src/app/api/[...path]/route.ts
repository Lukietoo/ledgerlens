// Runtime proxy: browser -> Next (/api/*) -> FastAPI. Replaces a build-time rewrite because
//  - the API's address is read per request (LEDGERLENS_API), so fixing it needs no rebuild;
//  - on Render's free tier the API is only reachable at its public URL (free services can't
//    receive private-network traffic) and sleeps when idle, so a cold start needs a long
//    timeout and a clear message instead of a generic proxy error;
//  - it forwards the visitor's IP, so the API's per-client rate limits see each visitor
//    rather than this server.
import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const TIMEOUT_MS = 70_000; // Render free-tier cold starts take ~30-60 s

function apiBase(): string {
  const raw = (process.env.LEDGERLENS_API ?? "http://127.0.0.1:8000").trim().replace(/\/+$/, "");
  return raw.includes("://") ? raw : `https://${raw}`;
}

function clientIp(req: NextRequest): string | undefined {
  // The hosting edge appends the address it saw as the LAST entry; earlier ones are client-supplied.
  const fwd = req.headers.get("x-forwarded-for");
  return fwd?.split(",").map((s) => s.trim()).filter(Boolean).pop() ?? req.headers.get("x-real-ip") ?? undefined;
}

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const target = `${apiBase()}/${path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  const headers = new Headers();
  const ct = req.headers.get("content-type");
  if (ct) headers.set("content-type", ct); // keeps the multipart boundary for CSV uploads
  const ip = clientIp(req);
  if (ip) headers.set("x-forwarded-for", ip);

  const hasBody = !["GET", "HEAD"].includes(req.method);
  try {
    const res = await fetch(target, {
      method: req.method,
      headers,
      body: hasBody ? await req.arrayBuffer() : undefined,
      signal: AbortSignal.timeout(TIMEOUT_MS),
      cache: "no-store",
    });
    const out = new Headers();
    for (const h of ["content-type", "retry-after"]) {
      const v = res.headers.get(h);
      if (v) out.set(h, v);
    }
    return new Response(res.body, { status: res.status, headers: out });
  } catch {
    return Response.json(
      { detail: "The API is waking up (free hosting sleeps when idle). Try again in about 30 seconds." },
      { status: 503, headers: { "retry-after": "30" } },
    );
  }
}

export { proxy as GET, proxy as POST, proxy as PUT, proxy as PATCH, proxy as DELETE };
