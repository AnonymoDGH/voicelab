//! `.vlvoice` files: a safetensors file with one tensor `spk_emb` (f32[256]) and string metadata.
//! Same idea as pocket-tts' exported voice states: the expensive encoder runs once per voice.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use crate::engine::vc::SPK_DIM;

pub const EXTENSION: &str = "vlvoice";

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
