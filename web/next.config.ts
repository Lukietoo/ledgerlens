import type { NextConfig } from "next";

// The browser talks only to Next. /api/* is proxied to the FastAPI service at request time by
// src/app/api/[...path]/route.ts (LEDGERLENS_API), so there's no CORS surface and the API's
// address can change without a rebuild.
const nextConfig: NextConfig = {};

export default nextConfig;
