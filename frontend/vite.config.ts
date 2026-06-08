import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built to ./dist, served by FastAPI at "/". API is same-origin under /api,
// proxied to the backend during `vite dev`.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
