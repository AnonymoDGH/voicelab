//! Low-latency pitch shifter: a read head plays a short delay line back at `2^(st/12)` times the
//! input speed. The delay therefore drifts; before it leaves its window a second head is
//! spawned one or more pitch periods away (where the waveform matches best, found by
//! normalized cross-correlation) and the two are crossfaded. Aligning the splice on the
//! waveform is what keeps voiced speech free of the warble of fixed two-head shifters.
//!
//! Formants move with the pitch (like tape speed), which suits the cartoon presets.
//! Every decision is taken per sample from the stream itself, so chunking cannot change the
//! output, and the delay stays below ~40 ms (average 15-25 ms, see `latency_samples`).

/// Crossfade between the dry input and the shifted signal when turning the shift on or off.
const BYPASS_FADE_S: f32 = 0.02;
/// Correlation window around the heads. It reaches mostly into what the crossfade will play:
/// on speech this picks splices almost as well as an oracle, while a window over the past only
/// loses ~0.2 of correlation. The future part must already be written, so it adds latency.
const WINDOW_PAST_S: f32 = 0.003;
const WINDOW_FUTURE_S: f32 = 0.008;
/// Jumps between heads. The search span covers a full period of an 80 Hz voice.
const JUMP_MIN_S: f32 = 0.010;
const JUMP_SPAN_S: f32 = 0.013;
/// Longest head crossfade; shorter for big shifts so it ends before the next jump is due.
const SPLICE_MAX_S: f32 = 0.010;
/// Smallest delay a head may have (cubic interpolation reads two samples ahead).
const MIN_DELAY: f64 = 3.0;
/// Shifts beyond this are clamped.
pub const MAX_SEMITONES: f32 = 24.0;
/// Below this the shift is treated as zero (exact bypass).
pub(super) const BYPASS_EPSILON: f32 = 1e-3;

/// Streaming pitch shifter, in place. At 0 semitones it is an exact, zero-latency bypass;
/// turning the shift on or off crossfades over 20 ms.
pub struct PitchShifter {
    sample_rate: u32,
    ring: Vec<f32>,
    mask: usize,
    /// Absolute index of the next sample to write.
    written: i64,
    semitones: f32,
    /// Playback speed of the active configuration (kept while fading out to bypass).
    ratio: f64,
    /// Read positions in absolute input samples; `cur` is the main head.
    heads: [f64; 2],
    cur: usize,
    splice_left: usize,
    splice_len: usize,
    // Derived from the sample rate.
    window_past: i64,
    window_future: i64,
    jump_min: i64,
    jump_max: i64,
    decimation: usize,
    max_delay: f64,
    // Derived from the ratio.
    splice: usize,
    trigger_up: f64,
    trigger_down: f64,
    /// 0 = dry input (bypass), 1 = shifted.
    wet: f32,
    wet_step: f32,
    /// Anti-aliasing low-pass (4th-order Butterworth) on what the heads read; matters when
    /// reading faster. Its cutoff glides to the target instead of jumping.
    aa: [Svf; 2],
    aa_cutoff: f32,
    aa_target: f32,
}

#[cfg(test)]
thread_local! {
    /// When enabled, every splice as (old head position, signed jump), for judging splice
    /// choices in tests.
    static SPLICE_LOG: std::cell::RefCell<Option<Vec<(i64, i64)>>> = const { std::cell::RefCell::new(None) };
}

/// Butterworth sections for a 4th-order low-pass.
const AA_Q: [f32; 2] = [0.5412, 1.3066];
/// Anti-aliasing cutoff relative to the sample rate, divided by the playback speed.
const AA_CUTOFF: f32 = 0.45;
/// The cutoff glides by at most this factor every `AA_GLIDE_EVERY` samples (~3 ms per octave
/// at 48 kHz): big retunes in one go would ring.
const AA_GLIDE_STEP: f32 = 1.06;
const AA_GLIDE_EVERY: i64 = 16;

/// Low-pass state-variable filter in the trapezoidal (TPT) form. Unlike a biquad, its states
/// are integrator outputs that stay valid when the cutoff jumps, so retuning it under audio
/// (switching presets swings the cutoff by 3x) does not leave a transient in the delay line.
#[derive(Clone, Debug)]
struct Svf {
    a1: f32,
    a2: f32,
    a3: f32,
    ic1: f32,
    ic2: f32,
}

impl Svf {
    fn new(sample_rate: u32, cutoff: f32, q: f32) -> Self {
        let mut f = Self { a1: 0.0, a2: 0.0, a3: 0.0, ic1: 0.0, ic2: 0.0 };
        f.set(sample_rate, cutoff, q);
        f
    }

    fn set(&mut self, sample_rate: u32, cutoff: f32, q: f32) {
        let cutoff = cutoff.clamp(1.0, 0.49 * sample_rate as f32);
        let g = (std::f64::consts::PI * cutoff as f64 / sample_rate as f64).tan();
        let k = 1.0 / q as f64;
        let a1 = 1.0 / (1.0 + g * (g + k));
        (self.a1, self.a2, self.a3) = (a1 as f32, (g * a1) as f32, (g * g * a1) as f32);
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let v3 = x - self.ic2;
        let v1 = self.a1 * self.ic1 + self.a2 * v3;
        let v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        v2
    }

    fn reset(&mut self) {
        (self.ic1, self.ic2) = (0.0, 0.0);
    }
}

impl PitchShifter {
    pub fn new(sample_rate: u32) -> Self {
        assert!(sample_rate > 0, "sample rate must be positive");
        let sr = sample_rate as f32;
        let samples = |s: f32| (s * sr).round() as i64;
        let (window_past, window_future) = (samples(WINDOW_PAST_S), samples(WINDOW_FUTURE_S));
        let jump_min = samples(JUMP_MIN_S);
        let jump_max = jump_min + samples(JUMP_SPAN_S);
        // Room for the longest delay (~42 ms at the extreme ratios) plus the window behind it.
        let ring_len = ((0.06 * sr) as usize + 64).next_power_of_two();
        let mut p = Self {
            sample_rate,
            ring: vec![0.0; ring_len],
            mask: ring_len - 1,
            written: 0,
            semitones: 0.0,
            ratio: 1.0,
            heads: [0.0; 2],
            cur: 0,
            splice_left: 0,
            splice_len: 1,
            window_past,
            window_future,
            jump_min,
            jump_max,
            decimation: ((sr / 8000.0).round() as usize).max(1),
            max_delay: (ring_len as i64 - window_past - 8) as f64,
            splice: 1,
            trigger_up: 0.0,
            trigger_down: 0.0,
            wet: 0.0,
            wet_step: 1.0 / (BYPASS_FADE_S * sr).max(1.0),
            aa: AA_Q.map(|q| Svf::new(sample_rate, AA_CUTOFF * sr, q)),
            aa_cutoff: AA_CUTOFF * sr,
            aa_target: AA_CUTOFF * sr,
        };
        p.configure(1.0);
        p
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn semitones(&self) -> f32 {
        self.semitones
    }

    /// Shift in semitones, clamped to ±[`MAX_SEMITONES`]. 0 crossfades to an exact bypass.
    pub fn set_semitones(&mut self, semitones: f32) {
        let st = if semitones.is_finite() { semitones.clamp(-MAX_SEMITONES, MAX_SEMITONES) } else { 0.0 };
        let st = if st.abs() < BYPASS_EPSILON { 0.0 } else { st };
        if st == self.semitones {
            return;
        }
        self.semitones = st;
        if st != 0.0 {
            self.configure(2f64.powf(st as f64 / 12.0));
            if self.wet == 0.0 {
                self.place_heads();
            }
        }
    }

    /// Average delay of the shifted signal; 0 when bypassed.
    pub fn latency_samples(&self) -> usize {
        if self.semitones == 0.0 {
            return 0;
        }
        let r = self.ratio;
        let jump = (self.jump_min + self.jump_max) as f64 / 2.0;
        // Delay at the middle of a head's life, crossfades included.
        let d = if r > 1.0 {
            self.trigger_up + jump / 2.0 - (r - 1.0) * self.splice as f64 / 2.0
        } else {
            self.trigger_down - jump / 2.0 + (1.0 - r) * self.splice as f64 / 2.0
        };
        d.round() as usize
    }

    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.written = 0;
        self.set_aa_cutoff(self.aa_target);
        for f in &mut self.aa {
            f.reset();
        }
        self.wet = if self.semitones != 0.0 { 1.0 } else { 0.0 };
        self.place_heads();
    }

    /// True when the output is the untouched input.
    pub fn is_bypassed(&self) -> bool {
        self.semitones == 0.0 && self.wet == 0.0
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        if self.is_bypassed() {
            // Keep the history current so turning the shift on fades in real audio.
            for &x in buf.iter() {
                self.write(x);
            }
            return;
        }
        let target = if self.semitones != 0.0 { 1.0 } else { 0.0 };
        for s in buf.iter_mut() {
            let x = if s.is_finite() { *s } else { 0.0 };
            self.write(x);
            let y = self.read_heads();
            self.wet += (target - self.wet).clamp(-self.wet_step, self.wet_step);
            *s = if self.wet == 1.0 { y } else { x + self.wet * (y - x) };
        }
    }

    fn configure(&mut self, ratio: f64) {
        self.ratio = ratio;
        let drift = (ratio - 1.0).abs().max(1e-9);
        let sr = self.sample_rate as f64;
        // The crossfade must finish before the new head drifts to the next trigger point.
        let splice = (SPLICE_MAX_S as f64 * sr).min(0.7 * self.jump_min as f64 / drift);
        self.splice = (splice as usize).max(1);
        let window_ok = (self.window_future + 1) as f64;
        // Reading faster: jump back once the delay gets small, early enough that the old head
        // can play out its crossfade, and while the window ahead of it is still written.
        self.trigger_up = (window_ok + (ratio - 1.0).max(0.0)).max(MIN_DELAY + drift * (self.splice + 1) as f64);
        // Reading slower: jump forward once even the longest jump lands at a usable delay.
        self.trigger_down = window_ok.max(MIN_DELAY) + self.jump_max as f64;
        self.aa_target = AA_CUTOFF * self.sample_rate as f32 / (ratio as f32).max(1.0);
    }

    fn set_aa_cutoff(&mut self, cutoff: f32) {
        self.aa_cutoff = cutoff;
        for (f, q) in self.aa.iter_mut().zip(AA_Q) {
            f.set(self.sample_rate, cutoff, q);
        }
    }

    /// Put the main head where a freshly spawned one would be.
    fn place_heads(&mut self) {
        let newest = (self.written - 1) as f64;
        let jump = ((self.jump_min + self.jump_max) / 2) as f64;
        let delay = if self.ratio > 1.0 { self.trigger_up + jump } else { self.trigger_down - jump };
        self.heads = [newest - delay; 2];
        self.cur = 0;
        self.splice_left = 0;
    }

    #[inline]
    fn write(&mut self, x: f32) {
        // Scheduled by sample count, so the glide does not depend on chunking.
        if self.aa_cutoff != self.aa_target && self.written % AA_GLIDE_EVERY == 0 {
            let ratio = self.aa_target / self.aa_cutoff;
            let next = if ratio > AA_GLIDE_STEP {
                self.aa_cutoff * AA_GLIDE_STEP
            } else if ratio < 1.0 / AA_GLIDE_STEP {
                self.aa_cutoff / AA_GLIDE_STEP
            } else {
                self.aa_target
            };
            self.set_aa_cutoff(next);
        }
        // Non-finite input would stay in the filter state for good.
        let x = if x.is_finite() { x } else { 0.0 };
        let x = self.aa.iter_mut().fold(x, |v, f| f.process(v));
        self.ring[self.written as usize & self.mask] = x;
        self.written += 1;
    }

    #[inline]
    fn read_heads(&mut self) -> f32 {
        let newest = (self.written - 1) as f64;
        if self.splice_left == 0 {
            let delay = newest - self.heads[self.cur];
            if self.ratio > 1.0 && delay <= self.trigger_up {
                self.start_splice(true);
            } else if self.ratio < 1.0 && delay >= self.trigger_down {
                self.start_splice(false);
            }
        }
        // Hard limits, only reachable when the ratio changes in the middle of a crossfade.
        for h in &mut self.heads {
            *h = h.clamp(newest - self.max_delay, newest - MIN_DELAY + 1.0);
        }
        let main = self.read(self.heads[self.cur]);
        let y = if self.splice_left > 0 {
            let other = self.read(self.heads[1 - self.cur]);
            let t = (self.splice_len - self.splice_left) as f32 + 0.5;
            let g = 0.5 - 0.5 * (std::f32::consts::PI * t / self.splice_len as f32).cos();
            self.splice_left -= 1;
            if self.splice_left == 0 {
                self.cur = 1 - self.cur;
            }
            main + g * (other - main)
        } else {
            main
        };
        self.heads[0] += self.ratio;
        self.heads[1] += self.ratio;
        y
    }

    fn start_splice(&mut self, backwards: bool) {
        let pos = self.heads[self.cur];
        let jump = self.best_jump(pos.floor() as i64, backwards) as f64;
        self.heads[1 - self.cur] = if backwards { pos - jump } else { pos + jump };
        self.splice_len = self.splice;
        self.splice_left = self.splice;
    }

    /// Jump (in samples) whose destination best matches the audio around `at`: a coarse search
    /// on a decimated grid, then a full-resolution refinement around the winner.
    fn best_jump(&self, at: i64, backwards: bool) -> i64 {
        let dir = if backwards { -1 } else { 1 };
        let window = -self.window_past..self.window_future;
        let search = |jumps: std::ops::RangeInclusive<i64>, step: usize| {
            let energy: f32 = window.clone().step_by(step).map(|m| self.at(at + m).powi(2)).sum();
            let mut best = (*jumps.start(), f32::NEG_INFINITY);
            for jump in jumps.step_by(step) {
                let (mut xy, mut yy) = (0.0f32, 0.0f32);
                for m in window.clone().step_by(step) {
                    let b = self.at(at + dir * jump + m);
                    xy += self.at(at + m) * b;
                    yy += b * b;
                }
                let score = xy / (energy * yy + 1e-12).sqrt();
                if score > best.1 {
                    best = (jump, score);
                }
            }
            (best.0, energy)
        };
        let step = self.decimation;
        let (coarse, energy) = search(self.jump_min..=self.jump_max, step);
        if energy < 1e-9 {
            // Silence: any jump works.
            return (self.jump_min + self.jump_max) / 2;
        }
        let lo = (coarse - step as i64 + 1).max(self.jump_min);
        let hi = (coarse + step as i64 - 1).min(self.jump_max);
        let j = search(lo..=hi, 1).0;
        #[cfg(test)]
        SPLICE_LOG.with(|l| {
            if let Some(log) = l.borrow_mut().as_mut() {
                log.push((at, if backwards { -j } else { j }));
            }
        });
        j
    }

    #[inline]
    fn at(&self, i: i64) -> f32 {
        // Negative indices wrap into the zero-initialized ring, i.e. silence before the start.
        self.ring[i as usize & self.mask]
    }

    /// Catmull-Rom cubic interpolation.
    #[inline]
    fn read(&self, pos: f64) -> f32 {
        let i = pos.floor();
        let t = (pos - i) as f32;
        let i = i as i64;
        let (xm1, x0, x1, x2) = (self.at(i - 1), self.at(i), self.at(i + 1), self.at(i + 2));
        let c1 = 0.5 * (x1 - xm1);
        let c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
        let c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
        ((c3 * t + c2) * t + c1) * t + x0
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{Lcg, chunked_in_place, dominant_freq, speech_16k};
    use super::*;
    use crate::dsp::resample::resample_clip;

    fn sine(sr: u32, f: f32, secs: f32, amp: f32) -> Vec<f32> {
        (0..(sr as f32 * secs) as usize)
            .map(|i| amp * (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin())
            .collect()
    }

    fn shifted(sr: u32, st: f32, x: &[f32]) -> Vec<f32> {
        let mut p = PitchShifter::new(sr);
        p.set_semitones(st);
        p.reset();
        let mut y = x.to_vec();
        p.process(&mut y);
        y
    }

    #[test]
    fn zero_semitones_is_an_exact_bypass() {
        let mut p = PitchShifter::new(16000);
        let x = speech_16k();
        let mut y = x.clone();
        p.process(&mut y);
        assert_eq!(x, y);
        assert_eq!(p.latency_samples(), 0);
        // Also after having been active: once the fade-out is over it is bit-exact again.
        p.set_semitones(5.0);
        p.process(&mut y);
        p.set_semitones(0.0);
        let mut y = x.clone();
        p.process(&mut y[..1000]);
        assert!(p.is_bypassed());
        let mut z = x.clone();
        p.process(&mut z);
        assert_eq!(x, z);
    }

    #[test]
    fn shifts_a_sine_accurately() {
        for sr in [16000, 48000] {
            for st in [12.0f32, -12.0, 3.0, -3.0, 1.0, 7.5] {
                let y = shifted(sr, st, &sine(sr, 220.0, 2.0, 0.5));
                let f = dominant_freq(&y[sr as usize / 4..], sr);
                let expected = 220.0 * 2f32.powf(st / 12.0);
                assert!((f / expected - 1.0).abs() < 0.02, "{sr} Hz, {st} st: {f:.1} Hz, expected {expected:.1}");
            }
        }
    }

    /// Worst signal-to-residual ratio (dB) of short segments against a sinusoid at `f` with
    /// free amplitude and phase per segment.
    fn worst_segment_snr(y: &[f32], sr: u32, f: f32) -> f64 {
        let w = 2.0 * std::f64::consts::PI * f as f64 / sr as f64;
        y.chunks_exact(sr as usize / 50)
            .map(|seg| {
                let (mut ss, mut sc, mut cc, mut ys, mut yc) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for (i, &v) in seg.iter().enumerate() {
                    let (s, c) = (w * i as f64).sin_cos();
                    (ss, sc, cc) = (ss + s * s, sc + s * c, cc + c * c);
                    (ys, yc) = (ys + v as f64 * s, yc + v as f64 * c);
                }
                let det = ss * cc - sc * sc;
                let (a, b) = ((ys * cc - yc * sc) / det, (yc * ss - ys * sc) / det);
                let (mut res, mut pow) = (0.0, 0.0);
                for (i, &v) in seg.iter().enumerate() {
                    let (s, c) = (w * i as f64).sin_cos();
                    res += (v as f64 - a * s - b * c).powi(2);
                    pow += (v as f64).powi(2);
                }
                10.0 * (pow / res.max(1e-30)).log10()
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn splices_keep_a_sine_clean() {
        // With waveform-aligned splices a shifted pure tone stays a pure tone in every 20 ms
        // segment. Fixed two-head shifters cancel the tone at each crossfade and fail this.
        for sr in [16000, 48000] {
            for st in [2.0f32, 4.0, -4.0, 12.0, -12.0] {
                let y = shifted(sr, st, &sine(sr, 180.0, 2.0, 0.5));
                let f = 180.0 * 2f32.powf(st / 12.0);
                let snr = worst_segment_snr(&y[sr as usize / 2..], sr, f);
                assert!(snr > 30.0, "{sr} Hz, {st} st: worst segment signal/residual {snr:.1} dB");
            }
        }
    }

    #[test]
    fn splices_on_speech_are_nearly_as_good_as_an_oracle() {
        // For every splice in periodic speech, compare how alike the two crossfaded heads are
        // with the best jump an oracle could pick knowing the audio being crossfaded.
        for sr in [16000u32, 48000] {
            let x = resample_clip(&speech_16k(), 16000, sr).unwrap();
            let (mut chosen_sum, mut oracle_sum, mut count, mut bad) = (0.0, 0.0, 0, 0);
            for st in [-4.0f32, -2.0, 2.0, 4.0, 9.0] {
                SPLICE_LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
                let mut p = PitchShifter::new(sr);
                p.set_semitones(st);
                p.reset();
                p.process(&mut x.clone());
                let (ratio, splice) = (p.ratio, p.splice);
                let similarity = |at: i64, jump: i64| {
                    let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
                    for k in 0..splice {
                        let i = at + (ratio * k as f64) as i64;
                        let (Some(&a), Some(&b)) = (x.get(i as usize), x.get((i + jump) as usize)) else { continue };
                        xy += a as f64 * b as f64;
                        xx += a as f64 * a as f64;
                        yy += b as f64 * b as f64;
                    }
                    (xy / (xx * yy).sqrt().max(1e-12), xx / splice as f64)
                };
                for (at, jump) in SPLICE_LOG.with(|l| l.borrow_mut().take().unwrap_or_default()) {
                    let (chosen, energy) = similarity(at, jump);
                    let oracle =
                        (p.jump_min..=p.jump_max).map(|j| similarity(at, j * jump.signum()).0).fold(f64::MIN, f64::max);
                    // Only audible, periodic material matters; noise cannot be aligned anyway.
                    if energy > 1e-5 && oracle > 0.8 {
                        chosen_sum += chosen;
                        oracle_sum += oracle;
                        count += 1;
                        bad += (chosen < 0.5) as usize;
                    }
                }
            }
            let (chosen, oracle) = (chosen_sum / count as f64, oracle_sum / count as f64);
            assert!(count >= 20, "{sr} Hz: only {count} periodic splices");
            assert!(chosen > oracle - 0.1 && chosen > 0.8, "{sr} Hz: similarity {chosen:.2}, oracle {oracle:.2}");
            assert!(bad <= count / 20, "{sr} Hz: {bad} of {count} splices badly aligned");
        }
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        for sr in [16000, 48000] {
            let x = resample_clip(&speech_16k(), 16000, sr).unwrap();
            for st in [-12.0f32, -3.5, 2.0, 12.0] {
                let mut p = PitchShifter::new(sr);
                p.set_semitones(st);
                let mut whole = x.clone();
                p.process(&mut whole);
                let mut p = PitchShifter::new(sr);
                p.set_semitones(st);
                let parts = chunked_in_place(&x, 3, |c| p.process(c));
                let err = whole.iter().zip(&parts).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(err <= 1e-5, "{sr} Hz {st} st: differs by {err}");
            }
        }
    }

    #[test]
    fn reported_latency_matches_impulse_delays() {
        for sr in [16000, 48000] {
            for st in [-12.0f32, -4.0, 4.0, 12.0] {
                let mut p = PitchShifter::new(sr);
                p.set_semitones(st);
                p.reset();
                let lat = p.latency_samples() as f32;
                let max = 0.05 * sr as f32;
                assert!(lat > 0.0 && lat < max, "{st} st: latency {lat} beyond 50 ms");
                // Impulses at irregular spacing land at every phase of the head cycle.
                let mut rng = Lcg(11);
                let mut x = vec![0.0f32; 4 * sr as usize];
                let mut at = Vec::new();
                let mut i = 1000;
                while i + sr as usize / 10 < x.len() {
                    x[i] = 1.0;
                    at.push(i);
                    i += sr as usize / 10 + rng.below(sr as usize / 20);
                }
                p.process(&mut x);
                let delays: Vec<f32> = at
                    .iter()
                    .map(|&i| {
                        let win = &x[i..i + max as usize];
                        win.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap().0 as f32
                    })
                    .collect();
                let mean = delays.iter().sum::<f32>() / delays.len() as f32;
                assert!(delays.iter().all(|&d| d < max), "{st} st: delay beyond 50 ms");
                assert!((mean - lat).abs() < 0.004 * sr as f32, "{sr} Hz {st} st: mean delay {mean}, reported {lat}");
            }
        }
    }

    #[test]
    fn switching_on_and_off_does_not_click() {
        let sr = 48000;
        let x = sine(sr, 150.0, 2.0, 0.8);
        let mut p = PitchShifter::new(sr);
        let mut y = x.clone();
        for (k, c) in y.chunks_mut(480).enumerate() {
            p.set_semitones(if (k / 20) % 2 == 1 { 5.0 } else { 0.0 });
            p.process(c);
        }
        // A 150 Hz sine at 0.8 moves at most ~0.016 per sample; shifted, ~0.021. Clicks would
        // be steps far above that.
        let max_step = y.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_step < 0.04, "max step {max_step}");
    }

    #[test]
    fn anti_alias_filter_retunes_without_transients() {
        // What lands in the delay line while the speed jumps between +21 st and neutral (the
        // cutoff swings 3.4x) under a 200 Hz tone, which both settings pass: it must stay as
        // smooth as the tone itself, or the heads would replay the transient later.
        let sr = 48000;
        let mut p = PitchShifter::new(sr);
        let (mut prev, mut max_step) = (0.0f32, 0.0f32);
        for i in 0..sr as usize {
            if i % 1000 == 0 {
                p.configure(if (i / 1000) % 2 == 0 { 3.4 } else { 1.0 });
            }
            p.write(0.5 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr as f32).sin());
            let y = p.at(p.written - 1);
            if i > 1000 {
                max_step = max_step.max((y - prev).abs());
            }
            prev = y;
        }
        // The tone itself moves by up to 2*pi*200/48000*0.5 = 0.0131 per sample; retuning in
        // one jump instead of gliding overshoots to ~0.042.
        assert!(max_step < 0.015, "max step {max_step}");
    }

    #[test]
    fn stays_finite_and_bounded() {
        let mut rng = Lcg(9);
        for st in [-24.0f32, -12.0, 0.5, 12.0, 24.0] {
            let mut p = PitchShifter::new(16000);
            p.set_semitones(st);
            let mut x: Vec<f32> = (0..32000).map(|i| if i % 500 < 250 { 1.0 } else { rng.signed() }).collect();
            x[100] = f32::NAN;
            p.process(&mut x);
            assert!(x.iter().all(|v| v.is_finite() && v.abs() < 2.0), "{st} st");
        }
    }
}
