//! Model directory layout (`manifest.json` written by tools/export) and ONNX session setup.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, serde::Serialize)]
pub enum Variant {
    /// 120 ms chunks + 40 ms lookahead: best quality, lowest CPU.
    #[serde(rename = "120ms")]
    Quality,
    /// 40 ms chunks + 40 ms lookahead: lower algorithmic latency, ~2x CPU.
    #[serde(rename = "40ms")]
    LowLatency,
    /// Quality with 4 mean-flow steps instead of 2, then AP-BWE up to 48 kHz.
    #[serde(rename = "ultra")]
    Ultra,
}

impl Variant {
    pub fn key(self) -> &'static str {
        match self {
            Variant::Quality => "120ms",
            Variant::LowLatency => "40ms",
            Variant::Ultra => "ultra",
        }
    }

    /// Ultra adds the 48 kHz bandwidth extension after the 16 kHz conversion.
    pub fn uses_bwe(self) -> bool {
        self == Variant::Ultra
    }
}

impl std::str::FromStr for Variant {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "120ms" | "quality" | "calidad" => Ok(Variant::Quality),
            "40ms" | "low-latency" | "baja-latencia" => Ok(Variant::LowLatency),
            "ultra" => Ok(Variant::Ultra),
            _ => bail!("unknown variant '{s}' (use 120ms, 40ms or ultra)"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VariantSpec {
    pub asr: String,
    pub dit: String,
    pub gtm: String,
    pub chunk_size: usize,
    pub block_size: usize,
    pub bn_window: usize,
    pub bn_stride: usize,
    pub asr_cache: usize,
    pub asr_offset_init: i64,
    pub asr_offset_step: i64,
    pub kv_shape: Vec<usize>,
    pub cache_frames: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VocoderSpec {
    pub file: String,
    pub n_fft: usize,
    pub hop: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BweSpec {
    pub file: String,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
    pub log_floor: f32,
    pub context_frames: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerSpec {
    pub file: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FileEntry {
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub sample_rate: u32,
    pub variants: BTreeMap<String, VariantSpec>,
    pub vocoder: VocoderSpec,
    pub speaker: SpeakerSpec,
    /// Absent in model sets from before Ultra mode.
    #[serde(default)]
    pub bwe: Option<BweSpec>,
    pub files: BTreeMap<String, FileEntry>,
}

/// A directory holding `manifest.json` and the ONNX graphs it lists.
#[derive(Debug, Clone)]
pub struct ModelDir {
    pub root: PathBuf,
    pub manifest: Manifest,
}

impl ModelDir {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let path = root.join("manifest.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let manifest: Manifest = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if manifest.format != 1 {
            bail!("unsupported model manifest format {}", manifest.format);
        }
        Ok(Self { root, manifest })
    }

    /// Whether this model set can run `v` (older downloads lack the Ultra graphs).
    pub fn supports(&self, v: Variant) -> bool {
        self.manifest.variants.contains_key(v.key()) && (!v.uses_bwe() || self.manifest.bwe.is_some())
    }

    pub fn variant(&self, v: Variant) -> Result<&VariantSpec> {
        self.manifest.variants.get(v.key()).with_context(|| format!("variant {} missing from manifest", v.key()))
    }

    pub fn path(&self, file: &str) -> PathBuf {
        self.root.join(file)
    }

    pub fn session(&self, file: &str, threads: usize) -> Result<Session> {
        let path = self.path(file);
        let cfg =
            |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow::anyhow!("configuring ONNX Runtime: {e}");
        Session::builder()
            .map_err(|e| anyhow::anyhow!("ONNX Runtime: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(cfg)?
            .with_intra_threads(threads.max(1))
            .map_err(cfg)?
            .with_inter_threads(1)
            .map_err(cfg)?
            .commit_from_file(&path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))
    }
}
