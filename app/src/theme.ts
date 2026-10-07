// Theme selection: "auto" follows the OS (Papel when light, Estudio when dark).

export interface ThemeInfo {
  id: string;
  name: string;
  hint: string;
}

export const THEMES: ThemeInfo[] = [
  { id: "auto", name: "Automático", hint: "Sigue el modo claro u oscuro de Windows" },
  { id: "estudio", name: "Estudio", hint: "Grafito cálido y ámbar, como un equipo de radio" },
  { id: "papel", name: "Papel", hint: "Claro, tinta y bermellón" },
  { id: "medianoche", name: "Medianoche", hint: "Azul pizarra y cian" },
  { id: "neon", name: "Neón", hint: "Negro, rosa eléctrico; para streaming" },
  { id: "contraste", name: "Alto contraste", hint: "Máxima legibilidad" },
];

const KEY = "voicelab.theme";
const light = window.matchMedia("(prefers-color-scheme: light)");
let current = "auto";

function resolve(id: string): string {
  if (id !== "auto") return id;
  return light.matches ? "papel" : "estudio";
}

function paint() {
  document.documentElement.dataset.theme = resolve(current);
}

export function applyTheme(id: string) {
  current = THEMES.some((t) => t.id === id) ? id : "auto";
  try {
    localStorage.setItem(KEY, current);
  } catch {
    // storage unavailable: the backend setting still persists the choice
  }
  paint();
}

export function initialTheme(): string {
  try {
    return localStorage.getItem(KEY) ?? "auto";
  } catch {
    return "auto";
  }
}

light.addEventListener("change", paint);
