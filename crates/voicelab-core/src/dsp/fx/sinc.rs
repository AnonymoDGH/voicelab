//! Sample-by-sample windowed-sinc resampler for the denoiser's trips to and from 48 kHz.
//!
//! The output clock is tracked as an exact rational position in the input (integer index plus
//! `frac / den`), so sample counts never drift and the output is a pure function of the input
//! sequence: chunking cannot change it. The filter is zero-phase (centred taps), so the signal
//! itself is not delayed; an output is emitted once its last tap has arrived, `half` input
//! samples later.

use std::f64::consts::PI;

/// Above this many distinct phases the table is quantized and adjacent phases interpolated.
const MAX_PHASES: u64 = 1024;
/// Sinc zero crossings on each side of the centre, at the lower of the two rates.
const ZERO_CROSSINGS: f64 = 24.0;
/// Pass band edge relative to the lower Nyquist frequency.
const CUTOFF: f64 = 0.9;

pub(super) struct SincResampler {
    /// Input samples advanced per output sample: `step_int + step_frac / den`.
    step_int: u64,
    step_frac: u64,
    den: u64,
    /// Position of the next output: input index `center` plus `frac / den`.
    center: u64,
    frac: u64,
    /// Number of input samples received so far.
    received: u64,
    half: usize,
    phases: u64,
    /// `phases + 1` rows of `2 * half` taps (the extra row is phase 0 shifted by one sample).
    table: Vec<f32>,
    /// The last `2 * half` inputs, written twice so the window is always contiguous.
    hist: Vec<f32>,
    hist_pos: usize,
}

impl SincResampler {
    pub fn new(from: u32, to: u32) -> Self {
        let g = gcd(from as u64, to as u64);
        let (num, den) = (from as u64 / g, to as u64 / g);
        // Normalized cutoff in input samples: the band shrinks when going down in rate.
        let cut = CUTOFF * (to as f64 / from as f64).min(1.0);
        let half = (ZERO_CROSSINGS / cut).ceil() as usize;
        let taps = 2 * half;
        let phases = den.min(MAX_PHASES);
        let mut table = vec![0.0f32; (phases as usize + 1) * taps];
        let mut tmp = vec![0.0f64; taps];
        for (p, row) in table.chunks_exact_mut(taps).enumerate() {
            let f = p as f64 / phases as f64;
            let mut sum = 0.0;
            for (m, t) in tmp.iter_mut().enumerate() {
                // Distance from this tap (input index center - half + 1 + m) to the output point.
                let x = f + (half - 1) as f64 - m as f64;
                *t = cut * sinc(cut * x) * blackman_harris(x / half as f64);
                sum += *t;
            }
            // Unity DC gain on every phase: otherwise the gain ripples at the output rate.
            for (r, t) in row.iter_mut().zip(&tmp) {
                *r = (t / sum) as f32;
            }
        }
        Self {
            step_int: num / den,
            step_frac: num % den,
            den,
            center: 0,
            frac: 0,
            received: 0,
            half,
            phases,
            table,
            hist: vec![0.0; 2 * taps],
            hist_pos: 0,
        }
    }

    /// Input samples an output waits for after the input instant it represents.
    pub fn lookahead(&self) -> usize {
        self.half
    }

    pub fn reset(&mut self) {
        self.center = 0;
        self.frac = 0;
        self.received = 0;
        self.hist.fill(0.0);
        self.hist_pos = 0;
    }

    /// Feed one input sample; `emit` receives every output that became computable.
    #[inline]
    pub fn push(&mut self, x: f32, mut emit: impl FnMut(f32)) {
        let taps = 2 * self.half;
        self.hist[self.hist_pos] = x;
        self.hist[self.hist_pos + taps] = x;
        self.hist_pos = (self.hist_pos + 1) % taps;
        self.received += 1;
        // Outputs are checked after every input, so a ready output's last tap is always the
        // newest sample and the window is exactly the history.
        while self.center + (self.half as u64) < self.received {
            let window = &self.hist[self.hist_pos..self.hist_pos + taps];
            let scaled = self.frac * self.phases;
            let (p, rem) = ((scaled / self.den) as usize, scaled % self.den);
            let row = |p: usize| &self.table[p * taps..(p + 1) * taps];
            let mut y = dot(row(p), window);
            if rem != 0 {
                let w = rem as f32 / self.den as f32;
                y += w * (dot(row(p + 1), window) - y);
            }
            emit(y);
            self.center += self.step_int;
            self.frac += self.step_frac;
            if self.frac >= self.den {
                self.frac -= self.den;
                self.center += 1;
            }
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Four accumulators let the compiler vectorize without reassociating a single sum.
    let mut acc = [0.0f32; 4];
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..4 {
            acc[k] += x[k] * y[k];
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 { 1.0 } else { (PI * x).sin() / (PI * x) }
}

/// 4-term Blackman-Harris over `u` in [-1, 1] (zero outside).
fn blackman_harris(u: f64) -> f64 {
    if u.abs() >= 1.0 {
        return 0.0;
    }
    let t = PI * (u + 1.0); // 0..2pi across the window
    0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos()
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(from: u32, to: u32, x: &[f32]) -> Vec<f32> {
        let mut r = SincResampler::new(from, to);
        let mut y = Vec::new();
        for &s in x {
            r.push(s, |v| y.push(v));
        }
        y
    }

    #[test]
    fn counts_follow_the_exact_ratio() {
        for (from, to) in
            [(16000, 48000), (48000, 16000), (44100, 48000), (48000, 44100), (22050, 48000), (37001, 48000)]
        {
            let n = 100_000usize;
            let r = SincResampler::new(from, to);
            let y = run(from, to, &vec![0.0; n]);
            let expected = (n - r.lookahead()) as f64 * to as f64 / from as f64;
            assert!((y.len() as f64 - expected).abs() <= 1.0, "{from}->{to}: {} vs {expected}", y.len());
        }
    }

    #[test]
    fn sine_is_resampled_cleanly_and_without_delay() {
        for (from, to) in [(16000, 48000), (48000, 16000), (44100, 48000), (48000, 22050), (37001, 48000)] {
            let f = 1000.0;
            let x: Vec<f32> = (0..from).map(|i| (2.0 * PI * f * i as f64 / from as f64).sin() as f32).collect();
            let y = run(from, to, &x);
            let err = (to as usize / 4..y.len() - 100)
                .map(|j| (y[j] as f64 - (2.0 * PI * f * j as f64 / to as f64).sin()).abs())
                .fold(0.0, f64::max);
            assert!(err < 2e-3, "{from}->{to}: max error {err}");
        }
    }

    #[test]
    fn rejects_content_above_the_target_band() {
        // 7.9 kHz at 48 kHz going down to 16 kHz sits in the transition band edge; 9 kHz must go.
        let x: Vec<f32> = (0..48000).map(|i| (2.0 * PI * 9000.0 * i as f64 / 48000.0).sin() as f32).collect();
        let y = run(48000, 16000, &x);
        let peak = y[4000..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.01, "alias peak {peak}");
    }
}
