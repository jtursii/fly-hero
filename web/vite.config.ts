import { defineConfig } from "vite";

export default defineConfig({
  // `web/public/` already holds the Phase 6 export (data/ + audio/); Vite
  // copies it verbatim into dist/, so paths are the same in dev and prod.
  build: { target: "es2022", outDir: "dist", assetsDir: "assets" },
  server: { port: 5173, open: false },
});
