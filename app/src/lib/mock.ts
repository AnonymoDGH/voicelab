// Browser-only stand-in for the Rust backend (UI preview and tests).
import type { Api, LiveStatus, Overview, Settings, VoiceInfo } from "./api";

const VCTK = "VCTK Corpus (CSTR, University of Edinburgh) via kyutai/tts-voices";

// Fingerprints of the built-in voices, as the backend computes them from their embeddings.
const REAL_PRINTS: Record<string, number[]> = {
  carlos: [0.411, 0.846, 0.62, 1.0, 0.846, 0.898, 0.38, 0.556, 0.79, 0.443, 0.517, 0.733, 0.15, 0.824, 0.603, 0.732, 0.476, 0.837, 0.463, 0.702, 0.624, 0.381, 0.347, 0.768, 0.548, 0.562, 0.544, 0.552, 0.378, 0.456, 0.504, 0.69],
  diego: [0.466, 0.724, 0.469, 0.864, 0.626, 0.967, 0.345, 0.178, 0.566, 0.277, 0.287, 0.468, 0.679, 0.508, 0.636, 0.325, 0.572, 0.323, 0.748, 0.927, 0.574, 0.539, 0.341, 0.525, 0.467, 1.0, 0.585, 0.496, 0.748, 0.583, 0.15, 0.387],
  elena: [1.0, 0.938, 0.36, 0.669, 0.602, 0.785, 0.455, 0.373, 0.313, 0.15, 0.811, 0.575, 0.622, 0.633, 0.624, 0.722, 0.526, 0.165, 0.718, 0.204, 0.624, 0.356, 0.271, 0.509, 0.682, 0.541, 0.412, 0.564, 0.672, 0.51, 0.723, 0.325],
  jorge: [0.692, 0.725, 0.15, 0.463, 0.454, 1.0, 0.961, 0.631, 0.504, 0.583, 0.438, 0.452, 0.541, 0.743, 0.692, 0.373, 0.818, 0.844, 0.606, 0.522, 0.655, 0.402, 0.483, 0.286, 0.77, 0.844, 0.503, 0.503, 0.532, 0.747, 0.772, 0.785],
  lucia: [0.655, 0.7, 1.0, 0.558, 0.465, 0.653, 0.212, 0.511, 0.378, 0.608, 0.78, 0.385, 0.84, 0.631, 0.977, 0.489, 0.15, 0.29, 0.494, 0.652, 0.456, 0.418, 0.468, 0.892, 0.506, 0.555, 0.785, 0.441, 0.354, 0.733, 0.582, 0.706],
  marta: [0.258, 0.652, 0.374, 0.47, 0.827, 0.796, 0.445, 0.513, 0.856, 0.52, 0.322, 0.556, 0.906, 0.851, 0.15, 0.412, 0.593, 0.439, 0.781, 0.957, 0.413, 0.916, 0.696, 0.679, 0.49, 0.759, 0.595, 1.0, 0.867, 0.675, 0.688, 0.649],
  pablo: [0.545, 0.31, 0.246, 1.0, 0.365, 0.513, 0.964, 0.175, 0.809, 0.499, 0.45, 0.419, 0.668, 0.327, 0.562, 0.665, 0.726, 0.332, 0.554, 0.777, 0.458, 0.641, 0.699, 0.552, 0.767, 0.15, 0.469, 0.319, 0.682, 0.43, 0.798, 0.47],
  sofia: [0.443, 0.589, 0.355, 0.15, 0.472, 0.878, 0.541, 0.879, 0.446, 0.231, 0.43, 0.552, 0.87, 0.879, 0.494, 0.482, 0.733, 1.0, 0.399, 0.303, 0.633, 0.607, 0.987, 0.642, 0.406, 0.688, 0.695, 0.568, 0.794, 0.302, 0.968, 0.588],
};

// Illustrations of the built-in voices, straight from the repo's voices/ folder.
const PORTRAITS = import.meta.glob<string>("../../../voices/*.svg", { query: "?url", import: "default", eager: true });
const portrait = (id: string) => PORTRAITS[`../../../voices/${id}.svg`] ?? null;

// Deterministic stand-in for voices without a known fingerprint.
function print(id: string): number[] {
  if (REAL_PRINTS[id]) return REAL_PRINTS[id];
  let h = 2166136261;
  for (const c of id) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
  return Array.from({ length: 32 }, () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0;
    return 0.15 + 0.85 * (h / 2 ** 32);
  });
}

export function mockApi(): Api {
  const voices: VoiceInfo[] = [
    ["carlos", "Carlos", "Masculina · VCTK p254"],
    ["diego", "Diego", "Masculina · VCTK p360"],
    ["elena", "Elena", "Femenina · VCTK p244"],
    ["jorge", "Jorge", "Masculina · VCTK p315"],
    ["lucia", "Lucía", "Femenina · VCTK p228"],
    ["marta", "Marta", "Femenina · VCTK p333"],
    ["pablo", "Pablo", "Masculina · VCTK p259"],
    ["sofia", "Sofía", "Femenina · VCTK p229"],
  ].map(([id, name, description]) => ({
    id,
    name,
    description,
    source: VCTK,
    license: "CC-BY-4.0",
    builtin: true,
    print: print(id),
    portrait: portrait(id),
    custom_portrait: false,
  }));
  const userVoice = (id: string, name: string, description: string, source: string): VoiceInfo => ({
    id,
    name,
    description,
    source,
    license: "propia",
    builtin: false,
    print: print(id),
    portrait: null,
    custom_portrait: false,
  });
  voices.push(userVoice("mi-voz", "Mi voz", "Grabada en casa", "grabación"));

  const params = new URLSearchParams(location.search);
  const settings: Settings = {
    input: null,
    output: null,
    monitor: null,
    monitor_enabled: false,
    variant: "40ms",
    threads: 1,
    voice: "carlos",
    gate_db: -50,
    input_gain: 1,
    output_gain: 1,
    hotkey: "CommandOrControl+Alt+V",
    theme: params.get("theme") ?? "estudio",
    denoise: true,
    effect: "none",
    pitch: 0,
    voice_hotkeys: true,
  };
  let running = false;
  let enabled = true;
  let muted = false;
  let modelsReady = params.get("models") !== "missing";
  const hasCable = params.get("cable") !== "missing";
  const t0 = performance.now();

  const overview = (): Overview => ({
    models_ready: modelsReady,
    models_dir: "C:\\Users\\tu\\AppData\\Roaming\\VoiceLab\\models",
    model_source: "https://github.com/AnonymoDGH/voicelab/releases/download/v0.1.0/",
    voices,
    inputs: [
      { id: "mic-1", name: "Micrófono (Realtek High Definition Audio)", is_default: true, is_virtual_cable: false },
      { id: "mic-2", name: "Micrófono USB (Blue Yeti)", is_default: false, is_virtual_cable: false },
    ],
    outputs: [
      { id: "out-1", name: "Altavoces (Realtek High Definition Audio)", is_default: true, is_virtual_cable: false },
      { id: "out-2", name: "Auriculares (USB Audio)", is_default: false, is_virtual_cable: false },
      ...(hasCable ? [{ id: "cable", name: "CABLE Input (VB-Audio Virtual Cable)", is_default: false, is_virtual_cable: true }] : []),
    ],
    virtual_cable: hasCable ? { id: "cable", name: "CABLE Input (VB-Audio Virtual Cable)", is_default: false, is_virtual_cable: true } : null,
    vb_cable_url: "https://vb-audio.com/Cable/",
    settings: { ...settings },
    running,
    info: running ? info() : null,
  });
  const info = () => ({
    input: "Micrófono (Realtek High Definition Audio)",
    output: hasCable ? "CABLE Input (VB-Audio Virtual Cable)" : "Altavoces (Realtek High Definition Audio)",
    monitor: "Auriculares (USB Audio)",
    output_is_virtual_cable: hasCable,
    input_rate: 48000,
    output_rate: 48000,
    block_ms: settings.variant === "40ms" ? 80 : 160,
    latency_ms: { "40ms": 225, "120ms": 305, ultra: 375 }[settings.variant],
  });
  const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

  return {
    overview: async () => overview(),
    async liveStatus(): Promise<LiveStatus> {
      const t = (performance.now() - t0) / 1000;
      const speech = Math.max(0, Math.sin(t * 2.1)) * (0.5 + 0.5 * Math.sin(t * 13.7));
      return {
        running,
        enabled,
        muted,
        error: null,
        stats: running
          ? {
              input_peak: 0.02 + 0.6 * speech,
              output_peak: muted ? 0 : enabled ? 0.5 * speech : 0.6 * speech,
              process_ms: 19 + 3 * Math.sin(t),
              process_ms_peak: 24,
              block_ms: settings.variant === "40ms" ? 80 : 160,
              load: { "40ms": 0.3, "120ms": 0.18, ultra: 0.46 }[settings.variant],
              output_buffer_ms: 70,
              gate_open: speech > 0.05,
              input_silent: params.get("mic") === "silent",
              blocks: Math.floor(t * 12),
              overruns: 0,
              underruns: 0,
              xruns: 0,
              errors: 0,
            }
          : null,
      };
    },
    async start() {
      if (!modelsReady) throw new Error("faltan los modelos: descárgalos en Ajustes");
      await wait(500);
      running = true;
      return info();
    },
    async stop() {
      running = false;
    },
    async setVoice(id) {
      settings.voice = id;
    },
    async setFlags(f) {
      if (f.enabled !== undefined) enabled = f.enabled;
      if (f.muted !== undefined) muted = f.muted;
      if (f.monitor !== undefined) settings.monitor_enabled = f.monitor;
    },
    async updateSettings(s) {
      Object.assign(settings, s);
    },
    async cloneFromFile(path, name, description) {
      await wait(1200);
      const v = userVoice(name.toLowerCase().replace(/\W+/g, "-"), name, description, path.split(/[\\/]/).pop() ?? "");
      voices.push(v);
      return v;
    },
    async recordAndClone(seconds, name, description) {
      await wait(seconds * 1000 + 1000);
      const v = userVoice(name.toLowerCase().replace(/\W+/g, "-"), name, description, "grabación");
      voices.push(v);
      return v;
    },
    async deleteVoice(id) {
      const i = voices.findIndex((v) => v.id === id);
      if (i >= 0) voices.splice(i, 1);
    },
    async setVoicePortrait(id, dataUrl) {
      const v = voices.find((v) => v.id === id)!;
      v.portrait = dataUrl;
      v.custom_portrait = true;
    },
    async clearVoicePortrait(id) {
      const v = voices.find((v) => v.id === id)!;
      v.portrait = v.builtin ? portrait(id) : null;
      v.custom_portrait = false;
    },
    async downloadModels(onProgress) {
      const total = 707_000_000;
      for (let done = 0; done <= total; done += 32_000_000) {
        onProgress({ done, total, file: done < total / 2 ? "spk_encoder.int8.onnx" : "dit_40ms.onnx" });
        await wait(60);
      }
      modelsReady = true;
    },
    pickAudioFile: async () => "C:\\Users\\tu\\Música\\mi_referencia.wav",
    openUrl: async (url) => void window.open(url, "_blank"),
    onEnabledChanged: async () => () => {},
    onVoiceChanged: async () => () => {},
  };
}
