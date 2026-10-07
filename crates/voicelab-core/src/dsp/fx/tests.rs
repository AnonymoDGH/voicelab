//! Tests of the whole chain; the processors have their own next to their code.

use super::test_util::{Lcg, allocations_during, chunked_in_place, dominant_freq, power_db, speech_16k};
use super::*;
use crate::dsp::resample::resample_clip;

fn params(effect: Effect, pitch_semitones: f32, mix: f32) -> FxParams {
    FxParams { effect, pitch_semitones, mix }
}

fn speech(sr: u32) -> Vec<f32> {
    resample_clip(&speech_16k(), 16000, sr).unwrap()
}

/// A chain already in the steady state of `p`.
fn chain(sr: u32, p: FxParams) -> FxChain {
    let mut c = FxChain::new(sr);
    c.set_params(p);
    c.reset();
    c
}

#[test]
fn passthrough_is_bit_exact() {
    let x = speech(48000);
    let mut c = FxChain::new(48000);
    let mut y = x.clone();
    c.process(&mut y);
    assert_eq!(x, y);
    assert_eq!(c.latency_samples(), 0);

    // Also once the fade back from an active effect is over.
    c.set_params(params(Effect::Cave, 3.0, 0.7));
    c.process(&mut y);
    c.set_params(FxParams::default());
    let mut fade = x[..2000].to_vec();
    c.process(&mut fade);
    let mut y = x.clone();
    c.process(&mut y);
    assert_eq!(x, y);
    assert_eq!(c.latency_samples(), 0);
}

#[test]
fn chunking_does_not_change_the_output() {
    for sr in [16000, 48000] {
        let x = speech(sr);
        for effect in Effect::ALL {
            for (pitch, mix) in [(0.0, 1.0), (2.5, 0.6)] {
                let p = params(effect, pitch, mix);
                // From the default state, so the switch-on fades are covered too.
                let mut a = FxChain::new(sr);
                a.set_params(p);
                let mut whole = x.clone();
                a.process(&mut whole);
                let mut b = FxChain::new(sr);
                b.set_params(p);
                let parts = chunked_in_place(&x, 42, |c| b.process(c));
                let err = whole.iter().zip(&parts).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(err <= 1e-5, "{effect:?} {pitch} st at {sr} Hz: differs by {err}");
            }
        }
    }
}

#[test]
fn loud_input_stays_bounded_and_finite() {
    let mut rng = Lcg(77);
    for sr in [16000, 48000] {
        let n = sr as usize * 2;
        let mut x: Vec<f32> = (0..n)
            .map(|i| match (i * 8 / n) % 3 {
                0 => {
                    if (i / 40) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                1 => rng.signed(),
                _ => 3.0 * (i as f32 * 0.05).sin(),
            })
            .collect();
        x[1234] = f32::NAN;
        x[2345] = f32::INFINITY;
        for effect in Effect::ALL {
            for pitch in [0.0f32, -12.0, 12.0] {
                if effect == Effect::None && pitch == 0.0 {
                    continue; // exact passthrough: garbage in, garbage out
                }
                let mut c = chain(sr, params(effect, pitch, 1.0));
                let mut y = x.clone();
                c.process(&mut y);
                let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(y.iter().all(|v| v.is_finite()), "{effect:?} {pitch} st: non-finite output");
                assert!(peak <= 1.0, "{effect:?} {pitch} st: peak {peak}");
            }
        }
    }
}

#[test]
fn switching_effects_adds_no_clicks() {
    // The largest sample-to-sample step while switching must not exceed what the effects
    // produce on their own in steady state. While the presets' pitch shift fades in or out,
    // a neighbouring effect briefly sees shifted input, so those combinations count too.
    let sr = 48000;
    let x: Vec<f32> =
        (0..sr as usize * 3).map(|i| 0.5 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr as f32).sin()).collect();
    let max_step = |y: &[f32]| y.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    let mut steady = 0.0f32;
    for effect in Effect::ALL {
        for pitch in [0.0, Effect::Demon.preset_semitones(), Effect::Chipmunk.preset_semitones()] {
            let mut c = chain(sr, params(effect, pitch, 1.0));
            let mut y = x.clone();
            c.process(&mut y);
            steady = steady.max(max_step(&y[sr as usize / 2..]));
        }
    }
    let mut c = FxChain::new(sr);
    let mut y = x.clone();
    let order = [
        Effect::Robot,
        Effect::Radio,
        Effect::None,
        Effect::Cave,
        Effect::Demon,
        Effect::Echo,
        Effect::Chipmunk,
        Effect::Robot,
    ];
    for (k, block) in y.chunks_mut(480).enumerate() {
        c.set_params(params(order[(k / 15) % order.len()], 0.0, 1.0));
        c.process(block);
    }
    let switching = max_step(&y);
    assert!(switching <= steady * 1.1, "switching step {switching} vs steady {steady}");
}

#[test]
fn presets_shift_the_pitch() {
    for sr in [16000, 48000] {
        let x: Vec<f32> = (0..sr as usize * 2)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin())
            .collect();
        for (effect, user) in
            [(Effect::Demon, 0.0f32), (Effect::Chipmunk, 0.0), (Effect::Chipmunk, -2.0), (Effect::Robot, 0.0)]
        {
            let mut c = chain(sr, params(effect, user, 1.0));
            let mut y = x.clone();
            c.process(&mut y);
            let expected = 220.0 * 2f32.powf((user + effect.preset_semitones()) / 12.0);
            if effect == Effect::Robot {
                // Ring modulation moves the energy to the sidebands 220 +- 75 Hz.
                let f = dominant_freq(&y[sr as usize / 2..], sr);
                assert!((f - 145.0).abs() < 3.0 || (f - 295.0).abs() < 3.0, "robot peak at {f}");
                continue;
            }
            let f = dominant_freq(&y[sr as usize / 2..], sr);
            assert!((f / expected - 1.0).abs() < 0.02, "{effect:?} at {sr} Hz: {f} vs {expected}");
            assert_eq!(c.latency_samples(), c.pitch.latency_samples());
            assert!(c.latency_samples() > 0);
        }
    }
}

#[test]
fn mix_blends_with_the_shifted_voice() {
    let sr = 16000;
    let x = speech(sr);
    let run = |mix: f32| {
        let mut c = chain(sr, params(Effect::Radio, 0.0, mix));
        let mut y = x.clone();
        c.process(&mut y);
        y
    };
    assert_eq!(run(0.0), x, "mix 0 leaves the voice (limiter aside, which speech does not reach)");
    let (half, full) = (run(0.5), run(1.0));
    let expected: Vec<f32> = x.iter().zip(&full).map(|(d, w)| 0.5 * (d + w)).collect();
    let err = half.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(err < 1e-5, "mix 0.5 is the average ({err})");
}

#[test]
fn params_round_trip_through_json() {
    let p = params(Effect::Chipmunk, -1.5, 0.25);
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(json, r#"{"effect":"chipmunk","pitch_semitones":-1.5,"mix":0.25}"#);
    assert_eq!(serde_json::from_str::<FxParams>(&json).unwrap(), p);
    let partial: FxParams = serde_json::from_str(r#"{"effect":"cave"}"#).unwrap();
    assert_eq!(partial, params(Effect::Cave, 0.0, 1.0));
    assert_eq!(serde_json::to_string(&Effect::None).unwrap(), r#""none""#);
}

#[test]
fn set_params_sanitizes() {
    let mut c = FxChain::new(16000);
    c.set_params(params(Effect::Echo, f32::NAN, 7.0));
    assert_eq!(c.params(), params(Effect::Echo, 0.0, 1.0));
    c.set_params(params(Effect::None, 99.0, -1.0));
    assert_eq!(c.params(), params(Effect::None, pitch::MAX_SEMITONES, 0.0));
}

#[test]
fn speech_levels_stay_reasonable() {
    let sr = 16000;
    let x = speech(sr);
    for effect in Effect::ALL {
        let mut c = chain(sr, params(effect, 0.0, 1.0));
        let mut y = x.clone();
        c.process(&mut y);
        let gain = power_db(&y) - power_db(&x);
        assert!(gain.abs() < 6.0, "{effect:?}: level changes by {gain:.1} dB");
    }
}

#[test]
fn works_at_every_common_rate() {
    for sr in [16000, 22050, 24000, 32000, 44100, 48000] {
        let x: Vec<f32> = speech(sr)[..sr as usize / 2].to_vec();
        for effect in Effect::ALL {
            let mut c = chain(sr, params(effect, 3.0, 1.0));
            let mut y = x.clone();
            c.process(&mut y);
            assert!(y.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "{effect:?} at {sr} Hz");
            assert!(power_db(&y) > power_db(&x) - 10.0, "{effect:?} at {sr} Hz is too quiet");
            assert!(c.latency_samples() < sr as usize / 20, "{effect:?} at {sr} Hz: latency over 50 ms");
        }
        let mut d = Denoiser::new(sr);
        let mut y = x.clone();
        d.process_in_place(&mut y);
        assert!(y.iter().all(|v| v.is_finite()), "denoiser at {sr} Hz");
        assert!(d.latency_samples() < sr as usize * 30 / 1000, "denoiser at {sr} Hz: latency over 30 ms");
    }
}

#[test]
fn processing_does_not_allocate() {
    for sr in [16000, 48000] {
        let x = speech(sr).repeat(2);
        let mut chain = FxChain::new(sr);
        let mut denoiser = Denoiser::new(sr);
        let (mut out, mut buf) = (Vec::with_capacity(10_000), vec![0.0f32; 10_000]);
        // nnnoiseless sets up FFT caches per thread on first use: warm it up on real speech.
        denoiser.process_in_place(&mut x[..sr as usize].to_vec());
        let mut rng = Lcg(13);
        let (mut chain_allocations, mut denoiser_allocations, mut pos) = (0, 0, 0);
        for k in 0.. {
            let n = rng.chunk_size();
            if pos + n > x.len() {
                break;
            }
            let buf = &mut buf[..n];
            buf.copy_from_slice(&x[pos..pos + n]);
            chain_allocations += allocations_during(|| {
                // Parameter changes every block, including effect and pitch switches.
                chain.set_params(params(Effect::ALL[(k / 3) % Effect::ALL.len()], (k % 5) as f32 - 2.0, 0.8));
                chain.process(buf);
            });
            denoiser_allocations += allocations_during(|| {
                out.clear();
                denoiser.process(buf, &mut out);
                denoiser.process_in_place(buf);
            });
            pos += n;
        }
        assert_eq!(chain_allocations, 0, "chain at {sr} Hz");
        assert_eq!(denoiser_allocations, 0, "denoiser at {sr} Hz");
    }
}

/// CPU cost per second of audio. Run with
/// `cargo test --release -p voicelab-core fx::tests::bench -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench() {
    use std::time::Instant;
    for sr in [16000u32, 48000] {
        let x = speech(sr);
        let secs = 20.0;
        let n = (sr as f32 * secs) as usize;
        let input: Vec<f32> = x.iter().copied().cycle().take(n).collect();
        // 10 ms blocks, like the engine.
        let report = |name: &str, f: &mut dyn FnMut(&mut [f32])| {
            let mut y = input.clone();
            let t = Instant::now();
            for c in y.chunks_mut(sr as usize / 100) {
                f(c);
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / secs as f64;
            println!("{sr:>5} Hz  {name:<28} {ms:7.3} ms CPU per second of audio");
        };
        for effect in Effect::ALL {
            for pitch in [0.0f32, 3.0] {
                let mut c = chain(sr, params(effect, pitch, 1.0));
                report(&format!("chain {effect:?} {pitch:+} st"), &mut |b| c.process(b));
            }
        }
        let mut p = PitchShifter::new(sr);
        p.set_semitones(-12.0);
        report("pitch -12 st", &mut |b| p.process(b));
        let mut d = Denoiser::new(sr);
        report("denoiser", &mut |b| d.process_in_place(b));

        let ms = |samples: usize| samples as f32 * 1000.0 / sr as f32;
        println!("{sr:>5} Hz  denoiser latency {:.1} ms", ms(d.latency_samples()));
        for st in [-12.0f32, -9.0, -4.0, -1.0, 1.0, 4.0, 9.0, 12.0] {
            let mut p = PitchShifter::new(sr);
            p.set_semitones(st);
            println!("{sr:>5} Hz  pitch {st:+} st latency {:.1} ms", ms(p.latency_samples()));
        }
    }
}
