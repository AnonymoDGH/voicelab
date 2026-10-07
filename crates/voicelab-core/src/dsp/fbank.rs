//! Kaldi log-mel filterbank, bit-compatible (to float rounding) with
//! `torchaudio.compliance.kaldi.fbank(frame_length=25, frame_shift=10, num_mel_bins=80,
//! snip_edges=True, dither=0, energy_floor=0)` at 16 kHz — the features Fast-U2++ was trained on.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};

pub const FRAME_LEN: usize = 400;
pub const FRAME_SHIFT: usize = 160;
pub const N_MELS: usize = 80;
const PADDED_LEN: usize = 512;
const N_BINS: usize = PADDED_LEN / 2 + 1;
const PREEMPH: f32 = 0.97;

/// One triangular filter, stored sparsely: weights for bins `start..start + weights.len()`.
struct MelFilter {
    start: usize,
    weights: Vec<f32>,
}

pub struct Fbank {
    window: Vec<f32>,
    filters: Vec<MelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
    frame: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    power: Vec<f32>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(PADDED_LEN);
        // Povey window: Hann (symmetric) raised to 0.85.
        let window = (0..FRAME_LEN)
            .map(|i| {
                let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_LEN - 1) as f64).cos();
                hann.powf(0.85) as f32
            })
            .collect();
        Self {
            window,
            filters: mel_filters(16000.0, 20.0),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            frame: vec![0.0; PADDED_LEN],
            power: vec![0.0; N_BINS],
            fft,
        }
    }

    /// Number of complete frames in `len` samples (snip_edges=True).
    pub fn num_frames(len: usize) -> usize {
        if len < FRAME_LEN { 0 } else { 1 + (len - FRAME_LEN) / FRAME_SHIFT }
    }

    /// Appends `num_frames(samples.len())` frames of 80 log-mel energies to `out`.
    /// `samples` are in [-1, 1]; Kaldi works on the int16 scale.
    pub fn compute(&mut self, samples: &[f32], out: &mut Vec<f32>) -> usize {
        let n = Self::num_frames(samples.len());
        out.reserve(n * N_MELS);
        for f in 0..n {
            let src = &samples[f * FRAME_SHIFT..f * FRAME_SHIFT + FRAME_LEN];
            let mean = src.iter().map(|&s| s as f64 * 32768.0).sum::<f64>() / FRAME_LEN as f64;
            let frame = &mut self.frame;
            for (d, &s) in frame.iter_mut().zip(src) {
                *d = (s as f64 * 32768.0 - mean) as f32;
            }
            // Pre-emphasis with replicate padding, applied back to front.
            for i in (1..FRAME_LEN).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];
            for (d, &w) in frame.iter_mut().zip(&self.window) {
                *d *= w;
            }
            frame[FRAME_LEN..].fill(0.0);
            self.fft.process_with_scratch(frame, &mut self.spectrum, &mut self.scratch).expect("fft sizes are fixed");
            for (p, c) in self.power.iter_mut().zip(&self.spectrum) {
                *p = c.norm_sqr();
            }
            for filter in &self.filters {
                let bins = &self.power[filter.start..filter.start + filter.weights.len()];
                let energy: f32 = bins.iter().zip(&filter.weights).map(|(p, w)| p * w).sum();
                out.push(energy.max(f32::EPSILON).ln());
            }
        }
        n
    }
}

fn mel_filters(sample_rate: f64, low_freq: f64) -> Vec<MelFilter> {
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let high_freq = sample_rate / 2.0;
    let bin_width = sample_rate / PADDED_LEN as f64;
    let (low_mel, high_mel) = (mel(low_freq), mel(high_freq));
    let delta = (high_mel - low_mel) / (N_MELS + 1) as f64;
    (0..N_MELS)
        .map(|b| {
            let left = low_mel + b as f64 * delta;
            let center = left + delta;
            let right = center + delta;
            // Kaldi uses bins 0..N/2 (the Nyquist bin gets weight 0).
            let weights: Vec<(usize, f32)> = (0..PADDED_LEN / 2)
                .filter_map(|k| {
                    let m = mel(bin_width * k as f64);
                    let w = ((m - left) / (center - left)).min((right - m) / (right - center));
                    (w > 0.0).then_some((k, w as f32))
                })
                .collect();
            let start = weights.first().map_or(0, |w| w.0);
            MelFilter { start, weights: weights.into_iter().map(|w| w.1).collect() }
        })
        .collect()
}

/// Dense `[80, 257]` filter matrix (for tests against torchaudio's `get_mel_banks`).
#[doc(hidden)]
pub fn dense_banks() -> Vec<f32> {
    let mut dense = vec![0.0; N_MELS * N_BINS];
    for (b, f) in mel_filters(16000.0, 20.0).iter().enumerate() {
        for (i, w) in f.weights.iter().enumerate() {
            dense[b * N_BINS + f.start + i] = *w;
        }
    }
    dense
}
