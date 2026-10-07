//! Real-time processor at device rates, without audio devices. Needs the models.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use voicelab_core::audio::processor::{Controls, Processor, Stats};
use voicelab_core::dsp::resample::resample_clip;
use voicelab_core::{ModelDir, StreamingVc, Variant, Voice};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn models() -> Option<ModelDir> {
    let dir = std::env::var("VOICELAB_MODELS").map(PathBuf::from).unwrap_or_else(|_| repo().join("models"));
    ModelDir::open(&dir).ok().or_else(|| {
        eprintln!("skipping: no models in {}", dir.display());
        None
    })
}

fn speech_48k() -> Vec<f32> {
    let f = std::fs::read(repo().join("tests/fixtures/golden_120ms.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&f).unwrap();
    let x: Vec<f32> =
        st.tensor("input").unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    resample_clip(&x, 16000, 48000).unwrap()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn run(p: &mut Processor, input: &[f32], sinks: usize) -> Vec<Vec<f32>> {
    let mut outs = vec![Vec::new(); sinks];
    // Irregular callback sizes, like real devices.
    let mut pos = 0;
    for (i, n) in [441usize, 480, 512, 1024, 333].iter().cycle().enumerate() {
        if pos >= input.len() || i > 100_000 {
            break;
        }
        let end = (pos + n).min(input.len());
        p.push(&input[pos..end], &mut outs).unwrap();
        pos = end;
    }
    outs
}

#[test]
fn converts_at_device_rates_with_gate_bypass_and_mute() {
    let Some(models) = models() else { return };
    let voice = Voice::load(repo().join("voices/carlos.vlvoice")).unwrap();
    let input = speech_48k();
    let controls = Arc::new(Controls::default());
    let stats = Arc::new(Stats::default());
    let new = |controls: &Arc<Controls>| {
        let mut vc = StreamingVc::new(&models, Variant::LowLatency, 1).unwrap();
        vc.set_speaker(&voice.embedding).unwrap();
        Processor::new(vc, 48000, &[48000, 44100], controls.clone(), stats.clone()).unwrap()
    };

    // Converted output on two sinks at their own rates, roughly as long as the input.
    let outs = run(&mut new(&controls), &input, 2);
    let secs = input.len() as f32 / 48000.0;
    for (out, rate) in outs.iter().zip([48000.0f32, 44100.0]) {
        let out_secs = out.len() as f32 / rate;
        assert!(out_secs > secs - 0.5 && out_secs <= secs + 0.05, "{rate}: {out_secs} s out for {secs} s in");
    }
    assert!(rms(&outs[0]) > 0.01, "converted speech should be audible, rms {}", rms(&outs[0]));
    assert!(stats.blocks.load(Ordering::Relaxed) > 10);

    // Silence + gate -> exact silence.
    let silence = vec![0.0f32; 48000];
    let outs = run(&mut new(&controls), &silence, 1);
    assert_eq!(rms(&outs[0]), 0.0, "gate must mute converted silence");

    // Bypass returns the input voice (through the 16 kHz path).
    controls.enabled.store(false, Ordering::Relaxed);
    let outs = run(&mut new(&controls), &input, 1);
    let ratio = rms(&outs[0]) / rms(&input);
    assert!((0.7..1.1).contains(&ratio), "bypass level ratio {ratio}");
    controls.enabled.store(true, Ordering::Relaxed);

    // Mute.
    controls.muted.store(true, Ordering::Relaxed);
    let outs = run(&mut new(&controls), &input, 1);
    assert_eq!(rms(&outs[0]), 0.0);
}
