//! Noise suppression with RNNoise (the pure-Rust `nnnoiseless` port).
//!
//! RNNoise only speaks 48 kHz in 480-sample frames, so other rates take a round trip through a
//! zero-phase windowed-sinc resampler. Everything runs sample by sample into a FIFO that was
//! pre-filled with enough zeros to cover the worst case of internal buffering, so every call
//! returns exactly as many samples as it got, the latency is constant and chunking cannot
//! change the output.
//!
//! Memory: all buffers are allocated in [`Denoiser::new`]. `nnnoiseless` caches its FFT plans
//! and scratch per thread, so the first frame processed on a new thread allocates once.

use std::sync::OnceLock;

use nnnoiseless::{DenoiseState, RnnModel};

use super::sinc::SincResampler;

const RNN_RATE: u32 = 48000;
const FRAME: usize = DenoiseState::FRAME_SIZE;
/// RNNoise's overlap-add returns each frame one frame late.
const RNN_DELAY: usize = FRAME;
/// RNNoise expects 16-bit PCM magnitudes in `f32`.
const PCM_SCALE: f32 = 32768.0;
/// Strength changes are ramped over this long to avoid zipper noise.
const STRENGTH_RAMP_S: f32 = 0.02;

/// The weights are parsed once and shared by every denoiser.
fn model() -> &'static RnnModel {
    static MODEL: OnceLock<RnnModel> = OnceLock::new();
    MODEL.get_or_init(RnnModel::default)
}

/// Streaming RNNoise denoiser with a dry/wet strength and constant latency (~20 ms at 48 kHz,
/// ~23 ms at 16 kHz).
pub struct Denoiser {
    sample_rate: u32,
    rnn: Box<DenoiseState<'static>>,
    /// To and from 48 kHz; `None` when already at 48 kHz.
    up: Option<SincResampler>,
    down: Option<SincResampler>,
    frame_in: Vec<f32>,
    frame_out: Vec<f32>,
    frame_fill: usize,
    first_frame: bool,
    /// Denoised audio at the input rate, waiting to be returned.
    wet: Fifo,
    /// Input delayed by the full latency, for the dry/wet mix.
    dry: Vec<f32>,
    dry_pos: usize,
    prefill: usize,
    latency: usize,
    strength: f32,
    /// Wet amount being applied, ramping towards `strength`.
    mix: f32,
    ramp_step: f32,
}

impl Denoiser {
    pub fn new(sample_rate: u32) -> Self {
        assert!(sample_rate > 0, "sample rate must be positive");
        let (up, down) = if sample_rate == RNN_RATE {
            (None, None)
        } else {
            (Some(SincResampler::new(sample_rate, RNN_RATE)), Some(SincResampler::new(RNN_RATE, sample_rate)))
        };
        let prefill = match (&up, &down) {
            (Some(up), Some(down)) => {
                // Worst-case shortfall of produced vs consumed samples (in input samples): the
                // upsampler's lookahead, a frame waiting to fill, and the downsampler's lookahead.
                // One extra sample absorbs the rounding of the rational positions.
                let s = sample_rate as f64 / RNN_RATE as f64;
                up.lookahead() + (s * (FRAME - 1 + down.lookahead()) as f64).ceil() as usize + 1
            }
            _ => FRAME - 1,
        };
        // The resamplers are zero-phase, so the wet signal lags by RNNoise's 10 ms only (to the
        // nearest sample where that is fractional, e.g. 220.5 samples at 22.05 kHz).
        let signal_delay = (RNN_DELAY as f64 * sample_rate as f64 / RNN_RATE as f64).round() as usize;
        let latency = prefill + signal_delay;
        let mut d = Self {
            sample_rate,
            rnn: DenoiseState::with_model(model()),
            up,
            down,
            frame_in: vec![0.0; FRAME],
            frame_out: vec![0.0; FRAME],
            frame_fill: 0,
            first_frame: true,
            wet: Fifo::new(prefill + 64),
            dry: vec![0.0; latency],
            dry_pos: 0,
            prefill,
            latency,
            strength: 1.0,
            mix: 1.0,
            ramp_step: 1.0 / (STRENGTH_RAMP_S * sample_rate as f32).max(1.0),
        };
        d.wet.fill_zeros(prefill);
        d
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Constant delay of the output relative to the input (both dry and wet parts).
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// 0 = untouched (but still delayed), 1 = fully denoised. Ramped over 20 ms.
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 1.0 };
    }

    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Append exactly `input.len()` samples to `output` (zeros while priming). Reuse `output`
    /// across calls so it stops growing.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.reserve(input.len());
        for &x in input {
            let y = self.tick(x);
            output.push(y);
        }
    }

    /// Same as [`Denoiser::process`], in place.
    pub fn process_in_place(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.tick(*s);
        }
    }

    /// Back to the freshly constructed state (keeps the strength). Allocates a new RNNoise state.
    pub fn reset(&mut self) {
        self.rnn = DenoiseState::with_model(model());
        for r in [&mut self.up, &mut self.down].into_iter().flatten() {
            r.reset();
        }
        self.frame_fill = 0;
        self.first_frame = true;
        self.wet.clear();
        self.wet.fill_zeros(self.prefill);
        self.dry.fill(0.0);
        self.dry_pos = 0;
        self.mix = self.strength;
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        // Non-finite input would poison RNNoise's recurrent state for good.
        let x = if x.is_finite() { x } else { 0.0 };
        let Self { rnn, up, down, frame_in, frame_out, frame_fill, first_frame, wet, .. } = self;
        let mut feed48 = |v: f32| {
            frame_in[*frame_fill] = v * PCM_SCALE;
            *frame_fill += 1;
            if *frame_fill < FRAME {
                return;
            }
            *frame_fill = 0;
            rnn.process_frame(frame_out, frame_in);
            if *first_frame {
                // The first frame covers time before the stream started and only carries the
                // network's start-up transient.
                *first_frame = false;
                frame_out.fill(0.0);
            }
            for &v in frame_out.iter() {
                let v = v / PCM_SCALE;
                match down {
                    Some(down) => down.push(v, |w| wet.push(w)),
                    None => wet.push(v),
                }
            }
        };
        match up {
            Some(up) => up.push(x, &mut feed48),
            None => feed48(x),
        }
        let w = wet.pop();

        let dry = std::mem::replace(&mut self.dry[self.dry_pos], x);
        self.dry_pos = (self.dry_pos + 1) % self.dry.len();

        self.mix += (self.strength - self.mix).clamp(-self.ramp_step, self.ramp_step);
        dry + self.mix * (w - dry)
    }
}

/// Fixed-capacity sample queue.
struct Fifo {
    buf: Vec<f32>,
    read: usize,
    len: usize,
}

impl Fifo {
    fn new(capacity: usize) -> Self {
        Self { buf: vec![0.0; capacity], read: 0, len: 0 }
    }

    fn clear(&mut self) {
        self.read = 0;
        self.len = 0;
    }

    fn fill_zeros(&mut self, n: usize) {
        for _ in 0..n {
            self.push(0.0);
        }
    }

    #[inline]
    fn push(&mut self, v: f32) {
        // The pre-fill bound keeps the queue below prefill + 3; dropping is a last resort.
        debug_assert!(self.len < self.buf.len(), "denoiser FIFO overflow");
        if self.len < self.buf.len() {
            let i = (self.read + self.len) % self.buf.len();
            self.buf[i] = v;
            self.len += 1;
        }
    }

    #[inline]
    fn pop(&mut self) -> f32 {
        debug_assert!(self.len > 0, "denoiser FIFO underflow");
        if self.len == 0 {
            return 0.0;
        }
        let v = self.buf[self.read];
        self.read = (self.read + 1) % self.buf.len();
        self.len -= 1;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{Lcg, chunked, power_db, speech_16k};
    use super::*;
    use crate::dsp::resample::resample_clip;

    fn run(d: &mut Denoiser, x: &[f32]) -> Vec<f32> {
        let mut y = Vec::new();
        d.process(x, &mut y);
        y
    }

    #[test]
    fn output_length_always_matches_input() {
        for sr in [16000, 22050, 44100, 48000] {
            let mut d = Denoiser::new(sr);
            let mut rng = Lcg(7);
            let mut out = Vec::new();
            let mut total = 0;
            for _ in 0..120 {
                let n = 1 + rng.below(700);
                let x: Vec<f32> = (0..n).map(|_| 0.1 * rng.signed()).collect();
                d.process(&x, &mut out);
                total += n;
                assert_eq!(out.len(), total);
            }
        }
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        for sr in [16000, 44100, 48000] {
            let x = resample_clip(&speech_16k(), 16000, sr).unwrap();
            let whole = run(&mut Denoiser::new(sr), &x);
            let mut d = Denoiser::new(sr);
            let parts = chunked(&x, 11, |c, out| d.process(c, out));
            let err = whole.iter().zip(&parts).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err <= 1e-5, "{sr} Hz: chunked output differs by {err}");
        }
    }

    #[test]
    fn zero_strength_is_a_pure_delay_of_the_reported_latency() {
        for sr in [16000, 48000] {
            let mut d = Denoiser::new(sr);
            d.set_strength(0.0);
            d.reset();
            let mut x = vec![0.0f32; sr as usize / 10];
            x[100] = 1.0;
            let y = run(&mut d, &x);
            let peak = y.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap();
            assert_eq!(peak.0, 100 + d.latency_samples(), "{sr} Hz");
            assert_eq!(*peak.1, 1.0);
        }
    }

    #[test]
    fn wet_path_is_aligned_with_the_reported_latency() {
        // Speech survives denoising, so the cross-correlation with the input peaks at the latency.
        for sr in [16000, 44100, 48000] {
            let x = resample_clip(&speech_16k(), 16000, sr).unwrap();
            let mut d = Denoiser::new(sr);
            let y = run(&mut d, &x);
            let lat = d.latency_samples();
            let best = (lat - 20..lat + 20)
                .max_by(|&a, &b| {
                    let c = |lag: usize| x.iter().zip(&y[lag..]).map(|(a, b)| a * b).sum::<f32>();
                    c(a).total_cmp(&c(b))
                })
                .unwrap();
            assert!(best.abs_diff(lat) <= 1, "{sr} Hz: correlation peak at {best}, reported {lat}");
        }
    }

    #[test]
    fn suppresses_white_noise_and_keeps_speech() {
        for sr in [16000, 48000] {
            let speech = resample_clip(&speech_16k(), 16000, sr).unwrap();
            let mut rng = Lcg(3);
            let noise: Vec<f32> = (0..speech.len()).map(|_| 0.03 * rng.signed()).collect();

            let mut d = Denoiser::new(sr);
            let y = run(&mut d, &noise);
            let skip = sr as usize / 2; // let the network converge
            let reduction = power_db(&noise[skip..]) - power_db(&y[skip..]);
            assert!(reduction > 10.0, "{sr} Hz: noise only reduced by {reduction:.1} dB");

            // Speech + noise: the output must stay close to the clean speech (SNR improves).
            let noisy: Vec<f32> = speech.iter().zip(&noise).map(|(s, n)| s + n).collect();
            let mut d = Denoiser::new(sr);
            let y = run(&mut d, &noisy);
            let lat = d.latency_samples();
            let snr = |out: &[f32], delay: usize| {
                let n = speech.len() - delay;
                let err: Vec<f32> = (skip..n).map(|i| out[i + delay] - speech[i]).collect();
                power_db(&speech[skip..n]) - power_db(&err)
            };
            let before = snr(&noisy, 0);
            let after = snr(&y, lat);
            assert!(after > before + 3.0, "{sr} Hz: SNR {before:.1} dB -> {after:.1} dB");
            // And clean speech is mostly kept.
            let mut d = Denoiser::new(sr);
            let y = run(&mut d, &speech);
            let kept = power_db(&y[lat..]) - power_db(&speech[..speech.len() - lat]);
            assert!(kept > -3.0, "{sr} Hz: clean speech lost {kept:.1} dB");
        }
    }

    #[test]
    fn strength_mixes_dry_and_wet() {
        let x = speech_16k();
        let mut rng = Lcg(5);
        let noisy: Vec<f32> = x.iter().map(|s| s + 0.03 * rng.signed()).collect();
        let mut outs = Vec::new();
        for s in [0.0, 0.5, 1.0] {
            let mut d = Denoiser::new(16000);
            d.set_strength(s);
            d.reset();
            outs.push(run(&mut d, &noisy));
        }
        let err = outs[0].iter().zip(&outs[1]).zip(&outs[2]).map(|((a, b), c)| (b - 0.5 * (a + c)).abs());
        assert!(err.fold(0.0f32, f32::max) < 1e-6, "half strength is the average of dry and wet");
    }

    #[test]
    fn survives_garbage_input() {
        let mut d = Denoiser::new(16000);
        let mut x = vec![4.0f32; 2000];
        x[10] = f32::NAN;
        x[20] = f32::INFINITY;
        let y = run(&mut d, &x);
        let y2 = run(&mut d, &vec![0.0; 4000]);
        assert!(y.iter().chain(&y2).all(|v| v.is_finite()));
    }
}
