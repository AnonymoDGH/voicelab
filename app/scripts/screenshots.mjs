// Regenerates the README images in ../docs/images from the UI running on the mock backend.
//
//   npx playwright install chromium   # once
//   npm run screenshots
//
// Images are encoded as WebP by Chromium itself (canvas.toDataURL), so no image tooling is needed.

import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const out = fileURLToPath(new URL("../../docs/images/", import.meta.url));
await mkdir(out, { recursive: true });

const server = await createServer({ root, server: { port: 5174, strictPort: true }, logLevel: "error" });
await server.listen();
const base = "http://localhost:5174";
const browser = await chromium.launch();
const errors = [];

async function page(path, { scale = 1, width = 1200, height = 800 } = {}) {
  const p = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: scale });
  p.on("pageerror", (e) => errors.push(`${path}: ${e}`));
  await p.goto(base + path);
  await p.evaluate(() => document.fonts.ready);
  return p;
}

const encoder = await browser.newPage();
async function save(name, png) {
  const dataUrl = await encoder.evaluate(async (b64) => {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);
    return c.toDataURL("image/webp", 0.92);
  }, png.toString("base64"));
  await writeFile(out + name, Buffer.from(dataUrl.split(",")[1], "base64"));
  console.log("wrote", name);
}

async function goLive(p) {
  await p.waitForSelector("text=Voces");
  await p.click(".power");
  await p.waitForSelector("text=En vivo");
  await p.keyboard.press("2"); // Carlos
  await p.waitForTimeout(1200);
  await p.evaluate(() => document.querySelector(".toast")?.remove());
}

// Banner
{
  const p = await page("/media/banner.html", { scale: 2, width: 1280, height: 440 });
  await save("banner.webp", await p.locator("#banner").screenshot());
}

// Main screenshot, live, default theme
{
  const p = await page("/?theme=estudio", { scale: 2 });
  await goLive(p);
  await save("app.webp", await p.screenshot());
}

// One per theme
for (const theme of ["estudio", "papel", "medianoche", "neon", "contraste"]) {
  const p = await page(`/?theme=${theme}`);
  await goLive(p);
  await save(`tema-${theme}.webp`, await p.screenshot());
}

// Clone flow
{
  const p = await page("/?theme=estudio");
  await p.waitForSelector("text=Voces");
  await p.click("nav >> text=Clonar");
  await p.click("text=Elegir audio");
  await p.fill("input[aria-label=Descripción]", "Grave, cálida");
  await p.check("input[type=checkbox]");
  await save("clonar.webp", await p.screenshot());
}

// Theme picker
{
  const p = await page("/?theme=estudio");
  await p.waitForSelector("text=Voces");
  await p.click("nav >> text=Ajustes");
  await p.waitForTimeout(300);
  await save("ajustes.webp", await p.screenshot());
}

await browser.close();
await server.close();
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
