//! Sample-rate conversion (rubato FFT resampler) for whole clips.

use anyhow::{Context, Result};
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

/// Resample a mono clip; returns the input unchanged when the rates match.
pub fn resample_clip(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let mut r = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Both).context("creating resampler")?;
    let input = InterleavedSlice::new(samples, 1, samples.len())?;
    let out = r.process_all(&input, samples.len(), None).context("resampling")?;
    Ok(out.take_data())
}
