// Bridge to the Rust backend. Outside Tauri (plain browser) a mock backend is used so the UI
// can be previewed and tested without audio hardware or models.

export type Variant = "120ms" | "40ms" | "ultra";

export type Effect = "none" | "robot" | "radio" | "demon" | "chipmunk" | "echo" | "cave";

export interface Settings {
  input: string | null;
  output: string | null;
  monitor: string | null;
  monitor_enabled: boolean;
  variant: Variant;
  threads: number;
  voice: string | null;
  gate_db: number;
  input_gain: number;
  output_gain: number;
  hotkey: string;
  theme: string;
  /** Noise suppression on the microphone. */
  denoise: boolean;
  effect: Effect;
  /** Pitch shift in semitones. */
  pitch: number;
  /** Ctrl+Alt+1..9 pick a voice from any app. */
  voice_hotkeys: boolean;
}

export interface Device {
  id: string;
  name: string;
  is_default: boolean;
  is_virtual_cable: boolean;
}

export interface VoiceInfo {
  id: string;
  name: string;
  description: string;
  source: string;
  license: string;
  builtin: boolean;
  /** 32 values in [0, 1] summarizing the speaker embedding (the voice's "fingerprint"). */
  print: number[];
}

export interface LiveInfo {
  input: string;
  output: string;
  monitor: string | null;
  output_is_virtual_cable: boolean;
  input_rate: number;
  output_rate: number;
  block_ms: number;
  latency_ms: number;
}

export interface Overview {
  models_ready: boolean;
  models_dir: string;
  model_source: string;
  voices: VoiceInfo[];
  inputs: Device[];
  outputs: Device[];
  virtual_cable: Device | null;
  vb_cable_url: string;
  settings: Settings;
  running: boolean;
  info: LiveInfo | null;
}

export interface Stats {
  input_peak: number;
  output_peak: number;
  process_ms: number;
  process_ms_peak: number;
  block_ms: number;
  load: number;
  output_buffer_ms: number;
  gate_open: boolean;
  /** Input has been exact digital silence for 3 s: wrong/muted mic or Windows privacy block. */
  input_silent: boolean;
  blocks: number;
  overruns: number;
  underruns: number;
  /** Glitches reported by Windows audio (not fatal). */
  xruns: number;
  errors: number;
}

export interface LiveStatus {
  running: boolean;
  enabled: boolean;
  muted: boolean;
  stats: Stats | null;
  error: string | null;
}

export interface DownloadProgress {
  done: number;
  total: number;
  file: string;
}

export interface Api {
  overview(): Promise<Overview>;
  liveStatus(): Promise<LiveStatus>;
  start(): Promise<LiveInfo>;
  stop(): Promise<void>;
  setVoice(id: string): Promise<void>;
  setFlags(flags: { enabled?: boolean; muted?: boolean; monitor?: boolean }): Promise<void>;
  updateSettings(settings: Settings): Promise<void>;
  cloneFromFile(path: string, name: string, description: string): Promise<VoiceInfo>;
  recordAndClone(seconds: number, name: string, description: string): Promise<VoiceInfo>;
  deleteVoice(id: string): Promise<void>;
  downloadModels(onProgress: (p: DownloadProgress) => void): Promise<void>;
  pickAudioFile(): Promise<string | null>;
  openUrl(url: string): Promise<void>;
  onEnabledChanged(cb: (enabled: boolean) => void): Promise<() => void>;
  /** A global voice hotkey switched the voice. */
  onVoiceChanged(cb: (id: string) => void): Promise<() => void>;
}

const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function tauriApi(): Promise<Api> {
  const { invoke } = await import("@tauri-apps/api/core");
  const { listen } = await import("@tauri-apps/api/event");
  const dialog = await import("@tauri-apps/plugin-dialog");
  const opener = await import("@tauri-apps/plugin-opener");
  return {
    overview: () => invoke("overview"),
    liveStatus: () => invoke("live_status"),
    start: () => invoke("start_engine"),
    stop: () => invoke("stop_engine"),
    setVoice: (id) => invoke("set_voice", { id }),
    setFlags: (f) => invoke("set_flags", { enabled: f.enabled ?? null, muted: f.muted ?? null, monitor: f.monitor ?? null }),
    updateSettings: (settings) => invoke("update_settings", { settings }),
    cloneFromFile: (path, name, description) => invoke("clone_voice_file", { path, name, description }),
    recordAndClone: (seconds, name, description) => invoke("record_and_clone", { seconds, name, description }),
    deleteVoice: (id) => invoke("delete_voice", { id }),
    async downloadModels(onProgress) {
      const unlisten = await listen<DownloadProgress>("download-progress", (e) => onProgress(e.payload));
      try {
        await invoke("download_models");
      } finally {
        unlisten();
      }
    },
    async pickAudioFile() {
      const file = await dialog.open({
        multiple: false,
        filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "ogg"] }],
      });
      return typeof file === "string" ? file : null;
    },
    openUrl: (url) => opener.openUrl(url),
    onEnabledChanged: (cb) => listen<boolean>("enabled-changed", (e) => cb(e.payload)),
    onVoiceChanged: (cb) => listen<string>("voice-changed", (e) => cb(e.payload)),
  };
}

export const api: Promise<Api> = inTauri ? tauriApi() : import("./mock").then((m) => m.mockApi());
