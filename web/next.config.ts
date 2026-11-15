import type { NextConfig } from "next";

// The browser talks only to Next; Next proxies /api/* to the FastAPI service,
// so there's no CORS surface and the API origin never ships to the client.
// Accepts a full URL or a bare host (Render's fromService "host" property has no scheme).
const raw = process.env.LEDGERLENS_API ?? "http://127.0.0.1:8000";
const API = raw.includes("://") ? raw : `https://${raw}`;

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
};

export default nextConfig;
