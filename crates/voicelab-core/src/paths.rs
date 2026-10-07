//! Where models and voices live.
//!
//! * Models: `$VOICELAB_MODELS`, else `<data>/models` (downloaded on first run), else `./models`.
//! * Voices: built-in `voices/` next to the executable (or in the repo when developing), plus
//!   the user's `<data>/voices`, where cloned voices are saved.
//!
//! `<data>` is `%APPDATA%\VoiceLab` on Windows and `~/.local/share/voicelab` on Linux.

use std::path::{Path, PathBuf};

pub fn data_dir() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    if cfg!(windows) { base.join("VoiceLab") } else { base.join("voicelab") }
}

pub fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("VOICELAB_MODELS") {
        return PathBuf::from(dir);
    }
    let user = data_dir().join("models");
    if user.join("manifest.json").exists() {
        return user;
    }
    let local = PathBuf::from("models");
    if local.join("manifest.json").exists() { local } else { user }
}

pub fn user_voices_dir() -> PathBuf {
    data_dir().join("voices")
}

/// Built-in voices directory: next to the executable when installed, `./voices` when developing.
pub fn builtin_voices_dir() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let candidates = exe_dir
        .iter()
        .flat_map(|d| [d.join("voices"), d.join("resources").join("voices")])
        .chain([PathBuf::from("voices")]);
    candidates.into_iter().find(|p| p.is_dir())
}

pub fn voice_dirs() -> Vec<PathBuf> {
    builtin_voices_dir().into_iter().chain([user_voices_dir()]).collect()
}
