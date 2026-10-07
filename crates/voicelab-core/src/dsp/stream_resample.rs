//! Streaming mono resampler: push any number of samples, get whatever output is ready.
//! Works in 10 ms input chunks, so every standard rate pair has an exact integer ratio.

use anyhow::{Context, Result};
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

pub struct StreamResampler {
    inner: Option<Fft<f32>>,
    pending: Vec<f32>,
    scratch: Vec<f32>,
}

impl StreamResampler {
    pub fn new(from: u32, to: u32) -> Result<Self> {
        let inner = if from == to {
            None
        } else {
            let chunk = (from / 100).max(1) as usize;
            Some(
                Fft::<f32>::new(from as usize, to as usize, chunk, 1, FixedSync::Input)
                    .context("creating resampler")?,
            )
        };
        let scratch = vec![0.0; inner.as_ref().map_or(0, |r| r.output_frames_max())];
        Ok(Self { inner, pending: Vec::new(), scratch })
    }

    /// Delay introduced by the resampler, in output samples.
    pub fn delay(&self) -> usize {
        self.inner.as_ref().map_or(0, |r| r.output_delay())
    }

    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let Some(r) = &mut self.inner else {
            out.extend_from_slice(input);
            return;
        };
        self.pending.extend_from_slice(input);
        let mut used = 0;
        loop {
            let need = r.input_frames_next();
            if self.pending.len() - used < need {
                break;
            }
            let n_out = r.output_frames_next();
            let src = InterleavedSlice::new(&self.pending[used..used + need], 1, need).expect("sizes checked");
            let mut dst = InterleavedSlice::new_mut(&mut self.scratch[..n_out], 1, n_out).expect("sizes checked");
            let (consumed, written) = r.process_into_buffer(&src, &mut dst, None).expect("valid buffers");
            out.extend_from_slice(&self.scratch[..written]);
            used += consumed;
        }
        self.pending.drain(..used);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_are_exact_over_time() {
        for (from, to) in [(48000, 16000), (44100, 16000), (16000, 48000), (16000, 44100), (16000, 16000)] {
            let mut r = StreamResampler::new(from, to).unwrap();
            let mut out = Vec::new();
            let block = vec![0.1f32; 777];
            let mut fed = 0usize;
            for _ in 0..200 {
                r.push(&block, &mut out);
                fed += block.len();
            }
            let expected = fed as f64 * to as f64 / from as f64;
            let slack = (from / 100 * 2) as f64 * to as f64 / from as f64;
            assert!((out.len() as f64 - expected).abs() <= slack, "{from}->{to}: {} vs {expected}", out.len());
        }
    }

    #[test]
    fn sine_survives_round_trip() {
        let mut down = StreamResampler::new(48000, 16000).unwrap();
        let mut up = StreamResampler::new(16000, 48000).unwrap();
        let x: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin()).collect();
        let (mut mid, mut y) = (Vec::new(), Vec::new());
        for c in x.chunks(512) {
            down.push(c, &mut mid);
        }
        up.push(&mid, &mut y);
        let d = down.delay() * 3 + up.delay();
        let err: f32 = (10000..30000).map(|i| (y[i] - x[i - d]).abs()).fold(0.0, f32::max);
        assert!(err < 0.05, "round trip error {err}");
    }
}
