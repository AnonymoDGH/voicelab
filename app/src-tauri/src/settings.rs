//! User settings, persisted as JSON in the data directory.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use voicelab_core::dsp::fx::Effect;
use voicelab_core::{Variant, paths};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub input: Option<String>,
    pub output: Option<String>,
    pub monitor: Option<String>,
    pub monitor_enabled: bool,
    pub variant: Variant,
    pub threads: usize,
    pub voice: Option<String>,
    pub gate_db: f32,
    pub input_gain: f32,
    pub output_gain: f32,
    pub hotkey: String,
    /// UI theme id ("auto", "estudio", "papel", "medianoche", "neon", "contraste").
    pub theme: String,
    /// Noise suppression on the microphone.
    pub denoise: bool,
    pub effect: Effect,
    /// Pitch shift in semitones.
    pub pitch: f32,
    /// Ctrl+Alt+1..9 pick the voice in that slot from any app.
    pub voice_hotkeys: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input: None,
            output: None,
            monitor: None,
            monitor_enabled: false,
            variant: Variant::LowLatency,
            threads: 1,
            voice: None,
            gate_db: -50.0,
            input_gain: 1.0,
            output_gain: 1.0,
            hotkey: "CommandOrControl+Alt+V".into(),
            theme: "auto".into(),
            denoise: false,
            effect: Effect::None,
            pitch: 0.0,
            voice_hotkeys: true,
        }
    }
}

fn path() -> PathBuf {
    paths::data_dir().join("settings.json")
}

impl Settings {
    pub fn load() -> Self {
        std::fs::read_to_string(path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }

    pub fn save(&self) {
        let p = path();
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(p, text);
        }
    }
}
