//! Second-order IIR filter with the RBJ "Audio EQ Cookbook" designs.
//!
//! Direct form I, because its state is plain past inputs and outputs: coefficients can change
//! while audio runs without the transient a transposed form emits when its internal state no
//! longer matches the new coefficients. The state is `f64` since the shelves and high-passes
//! used here sit at a few hundred Hz, where the poles crowd the unit circle at 48 kHz.

use std::f64::consts::PI;

/// Filter response. Gains are in dB; they only matter for the peaking and shelving types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterKind {
    Lowpass,
    Highpass,
    /// Constant 0 dB peak gain band-pass.
    Bandpass,
    Peaking {
        gain_db: f32,
    },
    LowShelf {
        gain_db: f32,
    },
    HighShelf {
        gain_db: f32,
    },
}

#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    /// Last two inputs and outputs.
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// `q` is the usual quality factor (0.707 = Butterworth); for shelves it sets the slope
    /// (0.707 is the steepest shelf without overshoot).
    pub fn new(kind: FilterKind, sample_rate: u32, freq: f32, q: f32) -> Self {
        let mut f = Self { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 };
        f.set(kind, sample_rate, freq, q);
        f
    }

    pub fn lowpass(sample_rate: u32, freq: f32, q: f32) -> Self {
        Self::new(FilterKind::Lowpass, sample_rate, freq, q)
    }

    pub fn highpass(sample_rate: u32, freq: f32, q: f32) -> Self {
        Self::new(FilterKind::Highpass, sample_rate, freq, q)
    }

    pub fn bandpass(sample_rate: u32, freq: f32, q: f32) -> Self {
        Self::new(FilterKind::Bandpass, sample_rate, freq, q)
    }

    pub fn peaking(sample_rate: u32, freq: f32, q: f32, gain_db: f32) -> Self {
        Self::new(FilterKind::Peaking { gain_db }, sample_rate, freq, q)
    }

    pub fn low_shelf(sample_rate: u32, freq: f32, q: f32, gain_db: f32) -> Self {
        Self::new(FilterKind::LowShelf { gain_db }, sample_rate, freq, q)
    }

    pub fn high_shelf(sample_rate: u32, freq: f32, q: f32, gain_db: f32) -> Self {
        Self::new(FilterKind::HighShelf { gain_db }, sample_rate, freq, q)
    }

    /// Recompute the coefficients, keeping the state (for parameter changes while running).
    /// The frequency is clamped below Nyquist so any setting stays stable at any rate.
    pub fn set(&mut self, kind: FilterKind, sample_rate: u32, freq: f32, q: f32) {
        let sr = sample_rate as f64;
        let freq = (freq as f64).clamp(1.0, 0.49 * sr);
        let w0 = 2.0 * PI * freq / sr;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * (q as f64).max(1e-3));
        let amp = |gain_db: f32| 10f64.powf(gain_db as f64 / 40.0);
        let (b0, b1, b2, a0, a1, a2) = match kind {
            FilterKind::Lowpass => {
                let b = (1.0 - cos) / 2.0;
                (b, 1.0 - cos, b, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
            }
            FilterKind::Highpass => {
                let b = (1.0 + cos) / 2.0;
                (b, -(1.0 + cos), b, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
            }
            FilterKind::Bandpass => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cos, 1.0 - alpha),
            FilterKind::Peaking { gain_db } => {
                let a = amp(gain_db);
                (1.0 + alpha * a, -2.0 * cos, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * cos, 1.0 - alpha / a)
            }
            FilterKind::LowShelf { gain_db } => {
                let a = amp(gain_db);
                let k = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) - (a - 1.0) * cos + k),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cos),
                    a * ((a + 1.0) - (a - 1.0) * cos - k),
                    (a + 1.0) + (a - 1.0) * cos + k,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cos),
                    (a + 1.0) + (a - 1.0) * cos - k,
                )
            }
            FilterKind::HighShelf { gain_db } => {
                let a = amp(gain_db);
                let k = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cos + k),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
                    a * ((a + 1.0) + (a - 1.0) * cos - k),
                    (a + 1.0) - (a - 1.0) * cos + k,
                    2.0 * ((a - 1.0) - (a + 1.0) * cos),
                    (a + 1.0) - (a - 1.0) * cos - k,
                )
            }
        };
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        (self.x2, self.x1) = (self.x1, x);
        (self.y2, self.y1) = (self.y1, y);
        y as f32
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    pub fn reset(&mut self) {
        (self.x1, self.x2, self.y1, self.y2) = (0.0, 0.0, 0.0, 0.0);
    }

    /// Magnitude response in dB at `freq` (analysis and tests).
    pub fn response_db(&self, sample_rate: u32, freq: f32) -> f32 {
        let w = 2.0 * PI * freq as f64 / sample_rate as f64;
        // H(e^jw) with z^-1 = e^-jw, evaluated as complex numbers by hand.
        let (c1, s1) = (w.cos(), -w.sin());
        let (c2, s2) = ((2.0 * w).cos(), -(2.0 * w).sin());
        let num = (self.b0 + self.b1 * c1 + self.b2 * c2, self.b1 * s1 + self.b2 * s2);
        let den = (1.0 + self.a1 * c1 + self.a2 * c2, self.a1 * s1 + self.a2 * s2);
        let mag2 = (num.0 * num.0 + num.1 * num.1) / (den.0 * den.0 + den.1 * den.1);
        (10.0 * mag2.log10()) as f32
    }
}

/// Run one sample through filters in series (e.g. two Butterworth sections for 4th order).
#[inline]
pub fn cascade(stages: &mut [Biquad], x: f32) -> f32 {
    stages.iter_mut().fold(x, |v, f| f.process_sample(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured_gain_db(f: &mut Biquad, sr: u32, freq: f32) -> f32 {
        let n = sr as usize;
        let x: Vec<f32> = (0..n).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin()).collect();
        let mut y = x.clone();
        f.process(&mut y);
        let rms = |v: &[f32]| (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt();
        // Skip the start: the filter state settles first.
        20.0 * (rms(&y[n / 2..]) / rms(&x[n / 2..])).log10()
    }

    #[test]
    fn cookbook_responses() {
        for sr in [16000, 48000] {
            let lp = Biquad::lowpass(sr, 1000.0, std::f32::consts::FRAC_1_SQRT_2);
            assert!((lp.response_db(sr, 1000.0) + 3.01).abs() < 0.05);
            assert!(lp.response_db(sr, 50.0).abs() < 0.05);
            assert!(lp.response_db(sr, 4000.0) < -20.0);

            let hp = Biquad::highpass(sr, 300.0, std::f32::consts::FRAC_1_SQRT_2);
            assert!((hp.response_db(sr, 300.0) + 3.01).abs() < 0.05);
            assert!(hp.response_db(sr, 60.0) < -25.0);

            let bp = Biquad::bandpass(sr, 1000.0, 2.0);
            assert!(bp.response_db(sr, 1000.0).abs() < 0.01);
            assert!(bp.response_db(sr, 200.0) < -15.0);

            let pk = Biquad::peaking(sr, 2000.0, 1.0, 6.0);
            assert!((pk.response_db(sr, 2000.0) - 6.0).abs() < 0.01);
            assert!(pk.response_db(sr, 100.0).abs() < 0.2);

            let ls = Biquad::low_shelf(sr, 200.0, 0.707, 8.0);
            assert!((ls.response_db(sr, 20.0) - 8.0).abs() < 0.2);
            assert!((ls.response_db(sr, 200.0) - 4.0).abs() < 0.1, "half gain at the corner");
            assert!(ls.response_db(sr, 5000.0).abs() < 0.1);

            let hs = Biquad::high_shelf(sr, 3000.0, 0.707, -6.0);
            assert!((hs.response_db(sr, 0.45 * sr as f32) + 6.0).abs() < 0.5);
            assert!(hs.response_db(sr, 100.0).abs() < 0.1);
        }
    }

    #[test]
    fn processing_matches_response() {
        let sr = 48000;
        let mut pk = Biquad::peaking(sr, 1000.0, 1.0, -9.0);
        let expected = pk.response_db(sr, 1000.0);
        assert!((measured_gain_db(&mut pk, sr, 1000.0) - expected).abs() < 0.05);
        let mut hp = Biquad::highpass(sr, 300.0, 0.707);
        let expected = hp.response_db(sr, 150.0);
        assert!((measured_gain_db(&mut hp, sr, 150.0) - expected).abs() < 0.05);
    }

    #[test]
    fn retuning_while_running_is_smooth() {
        // A 200 Hz tone through a low-pass retuned by 10% every 1000 samples, like a knob
        // being turned: the tone is in both pass bands, so the output must not jump.
        let sr = 48000;
        let mut lp = Biquad::lowpass(sr, 3000.0, 0.707);
        let mut prev = 0.0f32;
        let mut max_step = 0.0f32;
        for i in 0..sr as usize {
            if i % 1000 == 0 {
                let f = if (i / 1000) % 2 == 0 { 3300.0 } else { 3000.0 };
                lp.set(FilterKind::Lowpass, sr, f, 0.707);
            }
            let y = lp.process_sample(0.5 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr as f32).sin());
            if i > 1000 {
                max_step = max_step.max((y - prev).abs());
            }
            prev = y;
        }
        // The tone itself moves by up to 2*pi*200/48000*0.5 = 0.013 per sample.
        assert!(max_step < 0.016, "max step {max_step}");
    }

    #[test]
    fn frequency_is_clamped_below_nyquist() {
        let mut lp = Biquad::lowpass(16000, 20000.0, 0.707);
        let mut buf = vec![1.0f32; 1000];
        lp.process(&mut buf);
        assert!(buf.iter().all(|v| v.is_finite() && v.abs() < 2.0));
    }
}
