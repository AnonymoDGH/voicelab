// Turns a picked image into a voice portrait: center-cropped square, 256×256, as a data: URL.
// Done here so the backend only ever stores small thumbnails.

export const PORTRAIT_SIZE = 256;

export async function portraitFromFile(file: File): Promise<string> {
  let img: ImageBitmap;
  try {
    img = await createImageBitmap(file);
  } catch {
    throw new Error("No se pudo leer la imagen; prueba con un JPG, PNG o WebP.");
  }
  const side = Math.min(img.width, img.height);
  const c = document.createElement("canvas");
  c.width = c.height = PORTRAIT_SIZE;
  const ctx = c.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, PORTRAIT_SIZE, PORTRAIT_SIZE);
  img.close();
  // WebP where the webview can encode it; otherwise toDataURL falls back to PNG by itself.
  return c.toDataURL("image/webp", 0.86);
}

/** "Mi voz" → "MV", "Lucía" → "L": the monogram shown when a voice has no picture. */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  return words
    .slice(0, 2)
    .map((w) => [...w][0].toUpperCase())
    .join("");
}
