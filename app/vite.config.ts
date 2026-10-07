import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Tauri expects a fixed port and no screen clearing so Rust errors stay visible.
// The mock backend shows the built-in voices' portraits from ../voices.
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: { port: 5173, strictPort: true, watch: { ignored: ["**/src-tauri/**"] }, fs: { allow: [".", "../voices"] } },
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
});
