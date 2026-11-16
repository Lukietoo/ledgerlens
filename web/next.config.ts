import type { NextConfig } from "next";

// The browser talks only to Next. /api/* is proxied to the FastAPI service at request time by
// src/app/api/[...path]/route.ts (LEDGERLENS_API), so there's no CORS surface and the API's
// address can change without a rebuild.
//
// The container build sets NEXT_OUTPUT=standalone: a self-contained server.js with only the
// node_modules it uses, so the image ships without the full dependency tree.
const nextConfig: NextConfig = {
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
};

export default nextConfig;
