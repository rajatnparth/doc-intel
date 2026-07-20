import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The dev server PROXIES the API instead of the API allowing CORS.
// Deliberate: a permissive CORS block added "for dev" has a way of shipping.
// The engine's server config stays untouched; the browser sees one origin.
//
// VITE_API_TARGET (ui/.env.local) overrides the engine's address — port 8000
// is a popular squat (Docker holds it on the machine this was written on).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_TARGET || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/v1": target,
        "/health": target,
      },
    },
  };
});
