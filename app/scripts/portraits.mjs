// Generates the portraits of the built-in voices: one illustrated, fictional character per
// voice, saved as ../../voices/<id>.svg next to its .vlvoice.
//
//   npm run portraits
//
// The art is DiceBear's "Notionists" style by Zoish (CC0 1.0), drawn in black ink on white with
// a transparent background, so the UI can tint it with the theme colors. Each character is cast
// by hand (no randomness): rerunning the script gives the same files.

import { readFile, readdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createAvatar } from "@dicebear/core";
import { notionists } from "@dicebear/collection";

const dir = fileURLToPath(new URL("../../voices/", import.meta.url));

// Head and shoulders, the same framing for everyone. Gestures and shirt icons are left out.
const FRAME = { scale: 94, translateY: 3, gestureProbability: 0, bodyIconProbability: 0 };

// gender must match the voice's description ("Masculina …" / "Femenina …").
const CAST = {
  carlos: { gender: "Masculina", hair: "variant09", body: "variant14", eyes: "variant04", brows: "variant05", lips: "variant03", nose: "variant05", beard: "variant11" },
  diego: { gender: "Masculina", hair: "variant53", body: "variant17", eyes: "variant05", brows: "variant09", lips: "variant14", nose: "variant10" },
  jorge: { gender: "Masculina", hair: "variant49", body: "variant06", eyes: "variant04", brows: "variant02", lips: "variant22", nose: "variant03", beard: "variant02", glasses: "variant11" },
  pablo: { gender: "Masculina", hair: "variant06", body: "variant25", eyes: "variant05", brows: "variant12", lips: "variant23", nose: "variant14" },
  elena: { gender: "Femenina", hair: "variant28", body: "variant21", eyes: "variant04", brows: "variant04", lips: "variant23", nose: "variant01" },
  lucia: { gender: "Femenina", hair: "variant41", body: "variant12", eyes: "variant05", brows: "variant05", lips: "variant03", nose: "variant08" },
  marta: { gender: "Femenina", hair: "variant39", body: "variant05", eyes: "variant04", brows: "variant09", lips: "variant14", nose: "variant12", glasses: "variant08" },
  sofia: { gender: "Femenina", hair: "variant58", body: "variant16", eyes: "variant05", brows: "variant12", lips: "variant22", nose: "variant06" },
};

// String metadata of a .vlvoice (a safetensors file): 8-byte header length, then JSON.
async function metadata(path) {
  const bytes = await readFile(path);
  const len = Number(bytes.readBigUInt64LE(0));
  return JSON.parse(bytes.subarray(8, 8 + len).toString("utf8")).__metadata__ ?? {};
}

const ids = (await readdir(dir)).filter((f) => f.endsWith(".vlvoice")).map((f) => f.slice(0, -".vlvoice".length));
for (const id of ids) {
  const cast = CAST[id];
  if (!cast) throw new Error(`${id}.vlvoice has no character in CAST`);
  const { name = id, description = "" } = await metadata(`${dir}${id}.vlvoice`);
  const { gender, beard, glasses, ...parts } = cast;
  if (!description.startsWith(gender)) throw new Error(`${id}: cast as ${gender}, but the voice is «${description}»`);

  const options = Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, [v]]));
  const svg = createAvatar(notionists, {
    seed: id,
    ...FRAME,
    ...options,
    beard: beard ? [beard] : undefined,
    beardProbability: beard ? 100 : 0,
    glasses: glasses ? [glasses] : undefined,
    glassesProbability: glasses ? 100 : 0,
  }).toString();
  await writeFile(`${dir}${id}.svg`, svg);
  console.log(`wrote ${id}.svg  ${name} — ${description}  (${(svg.length / 1024).toFixed(1)} KB)`);
}
