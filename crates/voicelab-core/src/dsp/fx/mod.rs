//! Real-time voice effects: noise suppression, pitch shifting and classic presets.
//!
//! Every processor is streaming (any chunk size, state kept between calls, output independent
//! of how the stream is chunked), runs at any rate from 16 to 48 kHz, reports its latency and
//! does not allocate while processing. [`FxChain`] combines pitch shift and the presets behind
//! one serializable [`FxParams`]; the [`Denoiser`] is separate because it belongs before
//! anything else (and its latency is not optional).

pub mod biquad;
pub mod denoise;
pub mod effects;
pub mod pitch;
mod sinc;

#[cfg(test)]
mod test_util;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

pub use biquad::{Biquad, FilterKind};
pub use denoise::Denoiser;
pub use effects::{Cave, Echo, Radio, Robot};
pub use pitch::PitchShifter;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    #[default]
    None,
    /// Ring modulator into a metallic comb.
    Robot,
    /// Walkie-talkie: telephone band, overdrive, squelched hiss.
    Radio,
    /// 9 semitones down, heavy low end, dull top, some grit.
    Demon,
    /// 9 semitones up, thinned out.
    Chipmunk,
    /// 250 ms echo with darkening repeats.
    Echo,
    /// Large, dark reverb.
    Cave,
}

impl Effect {
    pub const ALL: [Effect; 7] =
        [Effect::None, Effect::Robot, Effect::Radio, Effect::Demon, Effect::Chipmunk, Effect::Echo, Effect::Cave];

    /// Pitch shift the preset adds on top of the user's.
    pub fn preset_semitones(self) -> f32 {
        match self {
            Effect::Demon => -9.0,
            Effect::Chipmunk => 9.0,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FxParams {
    pub effect: Effect,
    /// User pitch shift ("tono"), added to the preset's.
    pub pitch_semitones: f32,
    /// 0..1: how much of the effect stage is heard over the pitch-shifted voice. It does not
    /// undo the preset pitch shift of `Demon`/`Chipmunk`, only their tone shaping.
    pub mix: f32,
}

impl Default for FxParams {
    fn default() -> Self {
        Self { effect: Effect::None, pitch_semitones: 0.0, mix: 1.0 }
    }
}

impl FxParams {
    /// No effect and no shift: the chain is an exact passthrough.
    pub fn is_passthrough(&self) -> bool {
        self.effect == Effect::None && self.pitch_semitones == 0.0
    }

    fn sanitized(self) -> Self {
        let finite = |v: f32, default: f32| if v.is_finite() { v } else { default };
        let pitch = finite(self.pitch_semitones, 0.0).clamp(-pitch::MAX_SEMITONES, pitch::MAX_SEMITONES);
        Self {
            effect: self.effect,
            // Same dead zone as the shifter, so "passthrough" agrees with it.
            pitch_semitones: if pitch.abs() < pitch::BYPASS_EPSILON { 0.0 } else { pitch },
            mix: finite(self.mix, 1.0).clamp(0.0, 1.0),
        }
    }
}

/// Effect switches, mix changes and the limiter engaging are all ramped over this long.
const FADE_S: f32 = 0.02;
/// Internal block size: bounds the scratch buffers whatever the caller's chunk size.
const BLOCK: usize = 256;

/// Pitch shift (user + preset) followed by the selected effect, with click-free switching and
/// a gentle output limiter while anything is active.
pub struct FxChain {
    sample_rate: u32,
    params: FxParams,
    pitch: PitchShifter,
    stages: Stages,
    /// Current gain of each effect's output, indexed by `Effect as usize`.
    gains: [f32; Effect::ALL.len()],
    fade_step: f32,
    mix: f32,
    /// How much of the limiter is applied (0 when idle, ramped).
    engaged: f32,
    limiter: Limiter,
    dry: Vec<f32>,
    acc: Vec<f32>,
    tmp: Vec<f32>,
}

impl FxChain {
    pub fn new(sample_rate: u32) -> Self {
        assert!(sample_rate > 0, "sample rate must be positive");
        let params = FxParams::default();
        let mut gains = [0.0; Effect::ALL.len()];
        gains[params.effect as usize] = 1.0;
        Self {
            sample_rate,
            params,
            pitch: PitchShifter::new(sample_rate),
            stages: Stages::new(sample_rate),
            gains,
            fade_step: 1.0 / (FADE_S * sample_rate as f32).max(1.0),
            mix: params.mix,
            engaged: 0.0,
            limiter: Limiter::new(sample_rate),
            dry: vec![0.0; BLOCK],
            acc: vec![0.0; BLOCK],
            tmp: vec![0.0; BLOCK],
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn params(&self) -> FxParams {
        self.params
    }

    /// Cheap when nothing changed; changes are crossfaded, so it can be called every block.
    pub fn set_params(&mut self, params: FxParams) {
        let params = params.sanitized();
        if params == self.params {
            return;
        }
        self.params = params;
        self.pitch.set_semitones(params.pitch_semitones + params.effect.preset_semitones());
    }

    /// Delay of the output; only the pitch shifter adds any.
    pub fn latency_samples(&self) -> usize {
        self.pitch.latency_samples()
    }

    /// Jump to the steady state of the current parameters with all history cleared.
    pub fn reset(&mut self) {
        self.pitch.reset();
        for e in Effect::ALL {
            self.stages.reset(e);
        }
        self.gains = [0.0; Effect::ALL.len()];
        self.gains[self.params.effect as usize] = 1.0;
        self.mix = self.params.mix;
        self.engaged = if self.params.is_passthrough() { 0.0 } else { 1.0 };
        self.limiter.reset();
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        if self.is_idle() {
            // Still feed the shifter so it has fresh history when the shift is turned on.
            self.pitch.process(buf);
            return;
        }
        for s in buf.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        self.pitch.process(buf);
        for block in buf.chunks_mut(BLOCK) {
            self.process_block(block);
        }
        if self.is_idle() {
            self.limiter.reset();
        }
    }

    fn is_idle(&self) -> bool {
        self.params.is_passthrough()
            && self.pitch.is_bypassed()
            && self.engaged == 0.0
            && self.gains.iter().enumerate().all(|(i, &g)| g == if i == Effect::None as usize { 1.0 } else { 0.0 })
    }

    fn process_block(&mut self, buf: &mut [f32]) {
        let n = buf.len();
        let (dry, acc) = (&mut self.dry[..n], &mut self.acc[..n]);
        dry.copy_from_slice(buf);
        acc.fill(0.0);
        let step = self.fade_step;
        for e in Effect::ALL {
            let goal = if e == self.params.effect { 1.0 } else { 0.0 };
            let mut g = self.gains[e as usize];
            if g == 0.0 {
                if goal == 0.0 {
                    continue;
                }
                // Coming in from silence: start without the tail of its last use.
                self.stages.reset(e);
            }
            let out = &mut self.tmp[..n];
            out.copy_from_slice(dry);
            self.stages.process(e, out);
            for (a, &v) in acc.iter_mut().zip(out.iter()) {
                g += (goal - g).clamp(-step, step);
                *a += g * v;
            }
            self.gains[e as usize] = g;
        }
        let mix_goal = self.params.mix;
        let engaged_goal = if self.params.is_passthrough() { 0.0 } else { 1.0 };
        for ((s, &d), &a) in buf.iter_mut().zip(dry.iter()).zip(acc.iter()) {
            self.mix += (mix_goal - self.mix).clamp(-step, step);
            self.engaged += (engaged_goal - self.engaged).clamp(-step, step);
            let v = d + self.mix * (a - d);
            let limited = self.limiter.process(v);
            *s = v + self.engaged * (limited - v);
        }
    }
}

/// The per-effect processors, dispatched by [`Effect`].
struct Stages {
    robot: Robot,
    radio: Radio,
    demon: DemonTone,
    chipmunk: ChipmunkTone,
    echo: Echo,
    cave: Cave,
}

impl Stages {
    fn new(sample_rate: u32) -> Self {
        Self {
            robot: Robot::new(sample_rate),
            radio: Radio::new(sample_rate),
            demon: DemonTone::new(sample_rate),
            chipmunk: ChipmunkTone::new(sample_rate),
            echo: Echo::new(sample_rate),
            cave: Cave::new(sample_rate),
        }
    }

    fn process(&mut self, effect: Effect, buf: &mut [f32]) {
        match effect {
            Effect::None => {}
            Effect::Robot => self.robot.process(buf),
            Effect::Radio => self.radio.process(buf),
            Effect::Demon => self.demon.process(buf),
            Effect::Chipmunk => self.chipmunk.process(buf),
            Effect::Echo => self.echo.process(buf),
            Effect::Cave => self.cave.process(buf),
        }
    }

    fn reset(&mut self, effect: Effect) {
        match effect {
            Effect::None => {}
            Effect::Robot => self.robot.reset(),
            Effect::Radio => self.radio.reset(),
            Effect::Demon => self.demon.reset(),
            Effect::Chipmunk => self.chipmunk.reset(),
            Effect::Echo => self.echo.reset(),
            Effect::Cave => self.cave.reset(),
        }
    }
}

/// Tone stage of the demon preset (the pitch drop happens in the shifter): heavy low end, dull
/// top and some grit.
struct DemonTone {
    low: Biquad,
    high: Biquad,
}

impl DemonTone {
    const DRIVE: f32 = 2.5;
    /// Makes up for the level the darker top end takes away.
    const MAKEUP: f32 = 1.2;

    fn new(sample_rate: u32) -> Self {
        Self {
            low: Biquad::low_shelf(sample_rate, 250.0, 0.707, 7.0),
            high: Biquad::high_shelf(sample_rate, 3000.0, 0.707, -6.0),
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let v = self.high.process_sample(self.low.process_sample(*s));
            // Linear for small signals, saturating towards MAKEUP / DRIVE on peaks.
            *s = (Self::DRIVE * v).tanh() * (Self::MAKEUP / Self::DRIVE);
        }
    }

    fn reset(&mut self) {
        self.low.reset();
        self.high.reset();
    }
}

/// Tone stage of the chipmunk preset: thin it out so it reads as small.
struct ChipmunkTone {
    highpass: Biquad,
    presence: Biquad,
}

impl ChipmunkTone {
    fn new(sample_rate: u32) -> Self {
        Self {
            highpass: Biquad::highpass(sample_rate, 220.0, 0.707),
            presence: Biquad::peaking(sample_rate, 3000.0, 1.0, 3.0),
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.presence.process_sample(self.highpass.process_sample(*s));
        }
    }

    fn reset(&mut self) {
        self.highpass.reset();
        self.presence.reset();
    }
}

/// Peak limiter at -1 dBFS (1 ms attack, 150 ms release) with a soft clipper behind it for
/// whatever overshoots during the attack. Never exceeds 1.0.
struct Limiter {
    envelope: f32,
    attack: f32,
    release: f32,
}

impl Limiter {
    const THRESHOLD: f32 = 0.89;
    const KNEE: f32 = 0.9;

    fn new(sample_rate: u32) -> Self {
        let coef = |secs: f32| 1.0 - (-1.0 / (secs * sample_rate as f32)).exp();
        Self { envelope: 0.0, attack: coef(0.001), release: coef(0.15) }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let level = x.abs();
        let k = if level > self.envelope { self.attack } else { self.release };
        self.envelope += k * (level - self.envelope);
        let v = if self.envelope > Self::THRESHOLD { x * Self::THRESHOLD / self.envelope } else { x };
        let a = v.abs();
        if a <= Self::KNEE {
            v
        } else {
            let room = 1.0 - Self::KNEE;
            (Self::KNEE + room * ((a - Self::KNEE) / room).tanh()).copysign(v)
        }
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
    }
}
