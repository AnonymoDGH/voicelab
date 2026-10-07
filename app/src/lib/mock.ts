// Browser-only stand-in for the Rust backend (UI preview and tests).
import type { Api, LiveStatus, Overview, Settings, VoiceInfo } from "./api";

const VCTK = "VCTK Corpus (CSTR, University of Edinburgh) via kyutai/tts-voices";

// Fake fingerprint: smooth deterministic bars from the id (the backend derives them from the embedding).
function print(id: string): number[] {
  let h = 2166136261;
  for (const c of id) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
  const rnd = () => ((h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0) / 2 ** 32);
  const raw = Array.from({ length: 32 }, rnd);
  return raw.map((_, i) => 0.25 + 0.75 * ((raw[i] + raw[(i + 1) % 32] + raw[(i + 31) % 32]) / 3));
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
  ].map(([id, name, description]) => ({ id, name, description, source: VCTK, license: "CC-BY-4.0", builtin: true, print: print(id) }));
  voices.push({ id: "mi-voz", name: "Mi voz", description: "Grabada en casa", source: "grabación", license: "propia", builtin: false, print: print("mi-voz") });

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
    model_repo: "VoidWalkercero/voicelab-models",
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
    latency_ms: settings.variant === "40ms" ? 225 : 305,
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
              load: settings.variant === "40ms" ? 0.3 : 0.18,
              output_buffer_ms: 70,
              gate_open: speech > 0.05,
              blocks: Math.floor(t * 12),
              overruns: 0,
              underruns: 0,
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
      const id = name.toLowerCase().replace(/\W+/g, "-");
      const v = { id, name, description, source: path.split(/[\\/]/).pop() ?? "", license: "propia", builtin: false, print: print(id) };
      voices.push(v);
      return v;
    },
    async recordAndClone(seconds, name, description) {
      await wait(seconds * 1000 + 1000);
      const id = name.toLowerCase().replace(/\W+/g, "-");
      const v = { id, name, description, source: "grabación", license: "propia", builtin: false, print: print(id) };
      voices.push(v);
      return v;
    },
    async deleteVoice(id) {
      const i = voices.findIndex((v) => v.id === id);
      if (i >= 0) voices.splice(i, 1);
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
  };
}
