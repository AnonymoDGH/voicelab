//! `.vlvoice` files: a safetensors file with one tensor `spk_emb` (f32[256]) and string metadata.
//! Same idea as pocket-tts' exported voice states: the expensive encoder runs once per voice.
//!
//! A voice's portrait is an image sidecar with the same stem (`carlos.svg` next to
//! `carlos.vlvoice`). The user can give any voice their own picture; it is saved in the user
//! voices dir, since the built-in one may be read-only, and wins over the sidecar.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use crate::engine::vc::SPK_DIM;

pub const EXTENSION: &str = "vlvoice";

/// Portrait formats: file extension and MIME type. Earlier ones win if a voice has several.
pub const PORTRAIT_TYPES: [(&str, &str); 5] = [
    ("webp", "image/webp"),
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("svg", "image/svg+xml"),
];

/// Portraits are small thumbnails; anything bigger is not one.
pub const PORTRAIT_MAX_BYTES: u64 = 1 << 20;

#[derive(Debug, Clone, PartialEq)]
pub struct Portrait {
    pub path: PathBuf,
    /// Set by the user (it can be removed); otherwise it ships with the voice.
    pub custom: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Voice {
    /// File stem; stable identifier.
    pub id: String,
    pub name: String,
    pub description: String,
    pub source: String,
    pub license: String,
    #[serde(skip)]
    pub embedding: Vec<f32>,
    #[serde(skip)]
    pub path: Option<PathBuf>,
}

impl Voice {
    pub fn new(id: impl Into<String>, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            source: String::new(),
            license: String::new(),
            embedding,
            path: None,
        }
    }

    /// 32 values in [0.15, 1]: RMS of consecutive 8-dim groups of the embedding, normalized.
    /// A compact visual "fingerprint" of the voice for the UI.
    pub fn fingerprint(&self) -> Vec<f32> {
        let groups: Vec<f32> = self
            .embedding
            .chunks(SPK_DIM / 32)
            .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        let (lo, hi) = groups.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let span = (hi - lo).max(1e-6);
        groups.iter().map(|v| 0.15 + 0.85 * (v - lo) / span).collect()
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let (_, header) = SafeTensors::read_metadata(&bytes).context("invalid voice file")?;
        let st = SafeTensors::deserialize(&bytes).context("invalid voice file")?;
        let t = st.tensor("spk_emb").context("voice file has no spk_emb")?;
        ensure!(t.dtype() == Dtype::F32 && t.shape() == [SPK_DIM], "spk_emb must be f32[{SPK_DIM}]");
        let embedding = t.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let meta = header.metadata().clone().unwrap_or_default();
        let get = |k: &str| meta.get(k).cloned().unwrap_or_default();
        let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("voz").to_string();
        Ok(Self {
            name: if get("name").is_empty() { id.clone() } else { get("name") },
            description: get("description"),
            source: get("source"),
            license: get("license"),
            id,
            embedding,
            path: Some(path.to_path_buf()),
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes: Vec<u8> = self.embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![SPK_DIM], &bytes)?;
        let meta: HashMap<String, String> = [
            ("name", &self.name),
            ("description", &self.description),
            ("source", &self.source),
            ("license", &self.license),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
        let data = safetensors::serialize([("spk_emb", view)], Some(meta))?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, data).with_context(|| format!("writing {}", path.display()))
    }

    /// The voice's picture: the user's own in `user_dir`, else the one next to its file.
    pub fn portrait(&self, user_dir: &Path) -> Option<Portrait> {
        let own = find_portrait(user_dir, &self.id).map(|path| Portrait { path, custom: true });
        own.or_else(|| {
            let dir = self.path.as_deref()?.parent()?;
            find_portrait(dir, &self.id).map(|path| Portrait { path, custom: false })
        })
    }
}

fn find_portrait(dir: &Path, id: &str) -> Option<PathBuf> {
    PORTRAIT_TYPES.iter().map(|(ext, _)| dir.join(format!("{id}.{ext}"))).find(|p| p.is_file())
}

/// Ids are file stems; refuse anything that would point elsewhere.
fn check_id(id: &str) -> Result<()> {
    ensure!(Path::new(id).file_name() == Some(OsStr::new(id)), "invalid voice id «{id}»");
    Ok(())
}

/// Removes `<id>.<ext>` portraits from `dir`, except the one with extension `keep`.
fn remove_portraits(dir: &Path, id: &str, keep: Option<&str>) -> Result<()> {
    for (ext, _) in PORTRAIT_TYPES.iter().filter(|(ext, _)| Some(*ext) != keep) {
        let path = dir.join(format!("{id}.{ext}"));
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("removing {}", path.display())),
        }
    }
    Ok(())
}

/// Saves `bytes` (an image of type `ext`, see [`PORTRAIT_TYPES`]) as the user's own picture
/// for voice `id`, replacing the previous one.
pub fn set_portrait(user_dir: &Path, id: &str, bytes: &[u8], ext: &str) -> Result<PathBuf> {
    check_id(id)?;
    let ext = PORTRAIT_TYPES
        .iter()
        .map(|(e, _)| *e)
        .find(|e| e.eq_ignore_ascii_case(ext))
        .with_context(|| format!("unsupported portrait format «{ext}»"))?;
    ensure!(bytes.len() as u64 <= PORTRAIT_MAX_BYTES, "portrait too large ({} KB)", bytes.len() >> 10);
    std::fs::create_dir_all(user_dir)?;
    let path = user_dir.join(format!("{id}.{ext}"));
    std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    remove_portraits(user_dir, id, Some(ext))?;
    Ok(path)
}

/// Removes the user's own picture for voice `id`; a built-in one shows again.
pub fn clear_portrait(user_dir: &Path, id: &str) -> Result<()> {
    check_id(id)?;
    remove_portraits(user_dir, id, None)
}

/// All `.vlvoice` files in `dirs`, sorted by name. Unreadable files are skipped.
pub fn list_voices(dirs: &[PathBuf]) -> Vec<Voice> {
    let mut voices: Vec<Voice> = dirs
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == EXTENSION))
        .filter_map(|p| Voice::load(p).ok())
        .collect();
    voices.sort_by_key(|v| v.name.to_lowercase());
    voices
}

/// Make a file-name-safe id from a display name.
pub fn slug(name: &str) -> String {
    let s: String = name
        .to_lowercase()
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'ä' => 'a',
            'é' | 'è' | 'ë' => 'e',
            'í' | 'ì' | 'ï' => 'i',
            'ó' | 'ò' | 'ö' => 'o',
            'ú' | 'ù' | 'ü' => 'u',
            'ñ' => 'n',
            c if c.is_ascii_alphanumeric() => c,
            _ => '-',
        })
        .collect();
    let s = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    if s.is_empty() { "voz".into() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty directory under the system temp dir.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("voicelab-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn voice_in(dir: &Path, id: &str) -> Voice {
        let mut v = Voice::new(id, id, vec![0.0; SPK_DIM]);
        v.path = Some(dir.join(format!("{id}.{EXTENSION}")));
        v
    }

    #[test]
    fn user_portrait_wins_over_the_builtin_one() {
        let root = scratch("portrait-builtin");
        let (builtin, user) = (root.join("builtin"), root.join("user"));
        std::fs::create_dir_all(&builtin).unwrap();
        let carlos = voice_in(&builtin, "carlos");
        assert_eq!(carlos.portrait(&user), None);

        std::fs::write(builtin.join("carlos.svg"), "<svg/>").unwrap();
        let shipped = Portrait { path: builtin.join("carlos.svg"), custom: false };
        assert_eq!(carlos.portrait(&user), Some(shipped.clone()));

        let own = set_portrait(&user, "carlos", b"png", "PNG").unwrap();
        assert_eq!(own, user.join("carlos.png"));
        assert_eq!(carlos.portrait(&user), Some(Portrait { path: own, custom: true }));

        set_portrait(&user, "carlos", b"webp", "webp").unwrap();
        assert!(!user.join("carlos.png").exists(), "a new picture replaces the old one");
        assert_eq!(carlos.portrait(&user).unwrap().path, user.join("carlos.webp"));

        clear_portrait(&user, "carlos").unwrap();
        assert_eq!(carlos.portrait(&user), Some(shipped), "back to the shipped picture");
        clear_portrait(&user, "carlos").unwrap(); // nothing left to remove
        assert!(builtin.join("carlos.svg").exists(), "never touches the built-in dir");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn user_voice_portrait_and_format_order() {
        let user = scratch("portrait-user");
        let mine = voice_in(&user, "mi-voz");
        assert_eq!(mine.portrait(&user), None);
        std::fs::write(user.join("mi-voz.svg"), "<svg/>").unwrap();
        std::fs::write(user.join("mi-voz.jpeg"), "jpeg").unwrap();
        assert_eq!(mine.portrait(&user).unwrap().path, user.join("mi-voz.jpeg"), "raster before svg");
        std::fs::write(user.join("mi-voz.webp"), "webp").unwrap();
        assert_eq!(mine.portrait(&user), Some(Portrait { path: user.join("mi-voz.webp"), custom: true }));
        assert_eq!(voice_in(&user, "otra").portrait(&user), None, "matched by id only");
        clear_portrait(&user, "mi-voz").unwrap();
        assert_eq!(mine.portrait(&user), None);
        std::fs::remove_dir_all(user).unwrap();
    }

    #[test]
    fn set_portrait_rejects_bad_input() {
        let user = scratch("portrait-bad");
        assert!(set_portrait(&user, "../fuera", b"png", "png").is_err());
        assert!(set_portrait(&user, "", b"png", "png").is_err());
        assert!(set_portrait(&user, "x", b"gif", "gif").is_err());
        assert!(set_portrait(&user, "x", &vec![0; PORTRAIT_MAX_BYTES as usize + 1], "png").is_err());
        assert!(clear_portrait(&user, "..").is_err());
        assert_eq!(std::fs::read_dir(&user).unwrap().count(), 0, "nothing written");
        std::fs::remove_dir_all(user).unwrap();
    }
}
