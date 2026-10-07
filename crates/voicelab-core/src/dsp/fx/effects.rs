//! Classic voice effects. Each one is a small in-place processor with zero latency whose
//! output is the complete effect (the ambience effects keep the direct voice in it); the chain
//! blends it with the unprocessed signal.

use std::f32::consts::{PI, SQRT_2};

use super::biquad::{Biquad, cascade};

const BUTTERWORTH_Q: [f32; 2] = [0.5412, 1.3066];

fn samples(sample_rate: u32, secs: f32) -> usize {
    ((secs * sample_rate as f32).round() as usize).max(1)
}

/// Ring modulator through a short resonant comb: inharmonic sidebands plus a fixed buzzing
/// resonance that flattens the intonation, the classic sci-fi robot.
pub struct Robot {
    phase: f32,
    phase_inc: f32,
    comb: Vec<f32>,
    pos: usize,
}

impl Robot {
    const CARRIER_HZ: f32 = 75.0;
    /// Share of the ring-modulated signal; the rest keeps the voice intelligible.
    const MOD_MIX: f32 = 0.75;
    /// 5 ms comb: resonances every 200 Hz, a metallic tin-can colour.
    const COMB_S: f32 = 0.005;
    const COMB_FEEDBACK: f32 = 0.5;

    pub fn new(sample_rate: u32) -> Self {
        Self {
            phase: 0.0,
            phase_inc: Self::CARRIER_HZ / sample_rate as f32,
            comb: vec![0.0; samples(sample_rate, Self::COMB_S)],
            pos: 0,
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        // sin^2 averages 1/2, so sqrt(2) keeps the modulated part at the input level; the comb's
        // noise gain is 1/sqrt(1 - g^2).
        let norm = (1.0 - Self::COMB_FEEDBACK * Self::COMB_FEEDBACK).sqrt();
        for s in buf.iter_mut() {
            let carrier = (2.0 * PI * self.phase).sin();
            self.phase += self.phase_inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
            let v = *s + Self::MOD_MIX * (*s * carrier * SQRT_2 - *s);
            let c = v + Self::COMB_FEEDBACK * self.comb[self.pos];
            self.comb[self.pos] = c;
            self.pos = (self.pos + 1) % self.comb.len();
            *s = c * norm;
        }
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.comb.fill(0.0);
        self.pos = 0;
    }
}

/// Walkie-talkie: telephone band (300-3400 Hz, 4th order each side), a honky mid boost, soft
/// overdrive and a hiss that only opens while there is signal, like a squelched receiver.
pub struct Radio {
    highpass: [Biquad; 2],
    lowpass: [Biquad; 2],
    presence: Biquad,
    envelope: f32,
    attack: f32,
    release: f32,
    rng: u32,
}

impl Radio {
    const LOW_HZ: f32 = 300.0;
    const HIGH_HZ: f32 = 3400.0;
    const DRIVE: f32 = 4.0;
    const OUTPUT_GAIN: f32 = 0.7;
    /// Hiss level relative to the signal envelope.
    const HISS: f32 = 0.12;
    const SEED: u32 = 0x9E37_79B9;

    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        Self {
            highpass: BUTTERWORTH_Q.map(|q| Biquad::highpass(sample_rate, Self::LOW_HZ, q)),
            lowpass: BUTTERWORTH_Q.map(|q| Biquad::lowpass(sample_rate, Self::HIGH_HZ, q)),
            presence: Biquad::peaking(sample_rate, 1800.0, 0.9, 6.0),
            envelope: 0.0,
            attack: 1.0 - (-1.0 / (0.005 * sr)).exp(),
            release: 1.0 - (-1.0 / (0.15 * sr)).exp(),
            rng: Self::SEED,
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let x = *s;
            let level = x.abs();
            let k = if level > self.envelope { self.attack } else { self.release };
            self.envelope += k * (level - self.envelope);

            let mut v = cascade(&mut self.highpass, x);
            v = self.presence.process_sample(v);
            // Quiet passages gain sqrt(DRIVE), peaks flatten out at 1/sqrt(DRIVE).
            v = (Self::DRIVE * v).tanh() / Self::DRIVE.sqrt();
            v += Self::HISS * self.envelope * self.noise();
            v = cascade(&mut self.lowpass, v);
            *s = v * Self::OUTPUT_GAIN;
        }
    }

    pub fn reset(&mut self) {
        for f in self.highpass.iter_mut().chain(&mut self.lowpass) {
            f.reset();
        }
        self.presence.reset();
        self.envelope = 0.0;
        self.rng = Self::SEED;
    }

    /// Uniform noise in [-1, 1) from a xorshift32 generator: deterministic and allocation-free.
    #[inline]
    fn noise(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Feedback delay with a low-pass in the loop, so each repeat is darker than the last.
pub struct Echo {
    line: Vec<f32>,
    pos: usize,
    damp_state: f32,
    damp: f32,
}

impl Echo {
    const DELAY_S: f32 = 0.25;
    const FEEDBACK: f32 = 0.35;
    const WET: f32 = 0.3;
    const DAMP_HZ: f32 = 3000.0;

    pub fn new(sample_rate: u32) -> Self {
        Self {
            line: vec![0.0; samples(sample_rate, Self::DELAY_S)],
            pos: 0,
            damp_state: 0.0,
            damp: 1.0 - (-2.0 * PI * Self::DAMP_HZ / sample_rate as f32).exp(),
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let delayed = self.line[self.pos];
            self.damp_state += self.damp * (delayed - self.damp_state);
            self.line[self.pos] = *s + Self::FEEDBACK * self.damp_state;
            self.pos = (self.pos + 1) % self.line.len();
            *s += Self::WET * self.damp_state;
        }
    }

    pub fn reset(&mut self) {
        self.line.fill(0.0);
        self.pos = 0;
        self.damp_state = 0.0;
    }
}

/// Freeverb-style reverb (8 damped combs into 4 allpasses) tuned as a large, dark space:
/// stretched delays, long feedback, heavy damping, a pre-delay and a band-limited send.
pub struct Cave {
    send_hp: Biquad,
    send_lp: Biquad,
    predelay: Vec<f32>,
    predelay_pos: usize,
    combs: [Comb; 8],
    allpasses: [Allpass; 4],
}

impl Cave {
    /// Freeverb's tunings at 44.1 kHz; mutually prime-ish so the echoes do not line up.
    const COMB_TUNING: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
    const ALLPASS_TUNING: [usize; 4] = [556, 441, 341, 225];
    /// Delays 1.6x Freeverb's: a bigger room with sparser early echoes.
    const SIZE: f32 = 1.6;
    const FEEDBACK: f32 = 0.88;
    const DAMP: f32 = 0.45;
    const PREDELAY_S: f32 = 0.03;
    const SEND: f32 = 0.015;
    const DRY: f32 = 0.8;
    const WET: f32 = 2.0;

    pub fn new(sample_rate: u32) -> Self {
        let rate = sample_rate as f32 / 44100.0;
        let len = |n: usize, scale: f32| ((n as f32 * scale).round() as usize).max(1);
        Self {
            send_hp: Biquad::highpass(sample_rate, 150.0, 0.707),
            send_lp: Biquad::lowpass(sample_rate, 3500.0, 0.707),
            predelay: vec![0.0; samples(sample_rate, Self::PREDELAY_S)],
            predelay_pos: 0,
            combs: Self::COMB_TUNING.map(|n| Comb::new(len(n, Self::SIZE * rate))),
            // The allpasses only diffuse; stretching them too would smear the onsets.
            allpasses: Self::ALLPASS_TUNING.map(|n| Allpass::new(len(n, rate))),
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let send = self.send_lp.process_sample(self.send_hp.process_sample(*s)) * Self::SEND;
            let delayed = std::mem::replace(&mut self.predelay[self.predelay_pos], send);
            self.predelay_pos = (self.predelay_pos + 1) % self.predelay.len();
            let mut wet: f32 = self.combs.iter_mut().map(|c| c.process(delayed)).sum();
            for ap in &mut self.allpasses {
                wet = ap.process(wet);
            }
            *s = Self::DRY * *s + Self::WET * wet;
        }
    }

    pub fn reset(&mut self) {
        self.send_hp.reset();
        self.send_lp.reset();
        self.predelay.fill(0.0);
        self.predelay_pos = 0;
        for c in &mut self.combs {
            c.reset();
        }
        for a in &mut self.allpasses {
            a.reset();
        }
    }
}

/// Feedback comb with a one-pole low-pass in the loop (Freeverb's `comb`).
struct Comb {
    line: Vec<f32>,
    pos: usize,
    store: f32,
}

impl Comb {
    fn new(len: usize) -> Self {
        Self { line: vec![0.0; len], pos: 0, store: 0.0 }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let out = self.line[self.pos];
        self.store = out * (1.0 - Cave::DAMP) + self.store * Cave::DAMP;
        self.line[self.pos] = x + self.store * Cave::FEEDBACK;
        self.pos = (self.pos + 1) % self.line.len();
        out
    }

    fn reset(&mut self) {
        self.line.fill(0.0);
        self.pos = 0;
        self.store = 0.0;
    }
}

/// Schroeder allpass with gain 0.5 (Freeverb's `allpass`).
struct Allpass {
    line: Vec<f32>,
    pos: usize,
}

impl Allpass {
    fn new(len: usize) -> Self {
        Self { line: vec![0.0; len], pos: 0 }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let delayed = self.line[self.pos];
        self.line[self.pos] = x + delayed * 0.5;
        self.pos = (self.pos + 1) % self.line.len();
        delayed - x
    }

    fn reset(&mut self) {
        self.line.fill(0.0);
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{Lcg, chunked_in_place, power_db, speech_16k};
    use super::*;

    trait InPlace {
        fn run(&mut self, buf: &mut [f32]);
        fn clear(&mut self);
    }
    macro_rules! in_place {
        ($($t:ty),*) => {$(
            impl InPlace for $t {
                fn run(&mut self, buf: &mut [f32]) { self.process(buf) }
                fn clear(&mut self) { self.reset() }
            }
        )*};
    }
    in_place!(Robot, Radio, Echo, Cave);

    fn all(sr: u32) -> Vec<(&'static str, Box<dyn InPlace>)> {
        vec![
            ("robot", Box::new(Robot::new(sr))),
            ("radio", Box::new(Radio::new(sr))),
            ("echo", Box::new(Echo::new(sr))),
            ("cave", Box::new(Cave::new(sr))),
        ]
    }

    #[test]
    fn chunking_and_reset_do_not_change_the_output() {
        for sr in [16000, 48000] {
            let x = crate::dsp::resample::resample_clip(&speech_16k(), 16000, sr).unwrap();
            for ((name, mut a), (_, mut b)) in all(sr).into_iter().zip(all(sr)) {
                let mut whole = x.clone();
                a.run(&mut whole);
                let parts = chunked_in_place(&x, 5, |c| b.run(c));
                assert_eq!(whole, parts, "{name} at {sr} Hz");
                b.clear();
                let again = chunked_in_place(&x, 6, |c| b.run(c));
                assert_eq!(whole, again, "{name} at {sr} Hz after reset");
            }
        }
    }

    #[test]
    fn levels_stay_close_to_the_input() {
        let x = speech_16k();
        for (name, mut fx) in all(16000) {
            let mut y = x.clone();
            fx.run(&mut y);
            let gain = power_db(&y) - power_db(&x);
            assert!(gain.abs() < 6.0, "{name}: level changes by {gain:.1} dB");
        }
    }

    #[test]
    fn loud_input_stays_finite() {
        let mut rng = Lcg(1);
        let x: Vec<f32> = (0..96000).map(|i| if (i / 300) % 2 == 0 { 1.0 } else { -1.0 } * rng.next_f32()).collect();
        for sr in [16000, 48000] {
            for (name, mut fx) in all(sr) {
                let mut y = x.clone();
                fx.run(&mut y);
                let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(y.iter().all(|v| v.is_finite()) && peak < 8.0, "{name}: peak {peak}");
            }
        }
    }

    #[test]
    fn echo_repeats_after_250_ms_and_decays() {
        let sr = 16000;
        let mut e = Echo::new(sr);
        let mut x = vec![0.0f32; sr as usize * 2];
        x[0] = 1.0;
        e.process(&mut x);
        let d = sr as usize / 4;
        let energy = |k: usize| x[k * d..k * d + 100].iter().map(|v| v * v).sum::<f32>();
        assert!(energy(1) > 1e-3 && energy(2) < energy(1) && energy(3) < energy(2));
        assert!(x[1..d].iter().all(|&v| v == 0.0), "nothing before the first repeat");
    }

    #[test]
    fn radio_is_band_limited() {
        let sr = 48000;
        let tone = |f: f32| -> f32 {
            let mut r = Radio::new(sr);
            let mut x: Vec<f32> =
                (0..sr as usize).map(|i| 0.05 * (2.0 * PI * f * i as f32 / sr as f32).sin()).collect();
            r.process(&mut x);
            power_db(&x[sr as usize / 2..])
        };
        let mid = tone(1500.0);
        assert!(tone(100.0) < mid - 20.0);
        assert!(tone(8000.0) < mid - 20.0);
    }

    #[test]
    fn cave_has_a_long_tail() {
        let sr = 48000;
        let mut c = Cave::new(sr);
        let mut rng = Lcg(2);
        let mut x = vec![0.0f32; sr as usize * 3];
        for v in &mut x[..sr as usize / 10] {
            *v = 0.5 * rng.signed();
        }
        let burst = power_db(&x[..sr as usize / 10]);
        c.process(&mut x);
        let early = power_db(&x[sr as usize / 5..sr as usize * 2 / 5]) - burst;
        let tail = power_db(&x[sr as usize..sr as usize * 3 / 2]) - burst;
        let late = power_db(&x[sr as usize * 5 / 2..]) - burst;
        assert!(early > -25.0, "strong reverb right after the burst ({early:.1} dB)");
        assert!(tail > -45.0, "audible tail one second later ({tail:.1} dB)");
        // RT60 of 2-3 s: about 20-30 dB per second.
        let rate = (tail - late) / 1.5;
        assert!((15.0..35.0).contains(&rate), "decays at {rate:.1} dB/s");
    }
}
