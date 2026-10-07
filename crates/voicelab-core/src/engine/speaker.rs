//! Speaker encoder (WavLM-Large + ECAPA-TDNN, int8): reference audio -> 256-d embedding.

use anyhow::{Result, ensure};
use ort::session::Session;
use ort::value::TensorRef;

use super::models::ModelDir;
use super::vc::{SAMPLE_RATE, SPK_DIM};

/// Seconds of reference audio used; longer clips are trimmed (cost grows quadratically).
pub const MAX_SECONDS: usize = 20;
pub const MIN_SECONDS: f32 = 3.0;

pub struct SpeakerEncoder {
    session: Session,
}

impl SpeakerEncoder {
    pub fn new(models: &ModelDir, threads: usize) -> Result<Self> {
        Ok(Self { session: models.session(&models.manifest.speaker.file, threads)? })
    }

    /// `wav` is 16 kHz mono.
    pub fn embed(&mut self, wav: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            wav.len() as f32 >= MIN_SECONDS * SAMPLE_RATE as f32,
            "reference audio is too short: need at least {MIN_SECONDS} s of speech"
        );
        let wav = &wav[..wav.len().min(MAX_SECONDS * SAMPLE_RATE as usize)];
        let out = self
            .session
            .run(ort::inputs![
                "wav" => TensorRef::from_array_view(([1usize, wav.len()], wav)).map_err(|e| anyhow::anyhow!("{e}"))?
            ])
            .map_err(|e| anyhow::anyhow!("ONNX Runtime: {e}"))?;
        let emb = out["emb"].try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?.1.to_vec();
        ensure!(emb.len() == SPK_DIM, "unexpected embedding size {}", emb.len());
        Ok(emb)
    }
}
