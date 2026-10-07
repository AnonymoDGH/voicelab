//! Parity with the Python reference (tools/export/make_golden.py fixtures).
//! Pipeline tests need the exported models: set VOICELAB_MODELS or put them in <repo>/models.

use std::path::PathBuf;

use safetensors::SafeTensors;
use voicelab_core::dsp::fbank::{Fbank, dense_banks};
use voicelab_core::dsp::istft::Istft;
use voicelab_core::engine::Bwe;
use voicelab_core::engine::vc::{StreamingVc, Trace};
use voicelab_core::{ModelDir, SpeakerEncoder, Variant};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Fixture(Vec<u8>);

impl Fixture {
    fn load(name: &str) -> Self {
        Self(std::fs::read(repo().join("tests/fixtures").join(name)).expect("fixture"))
    }
    fn f32(&self, key: &str) -> Vec<f32> {
        let st = SafeTensors::deserialize(&self.0).unwrap();
        let t = st.tensor(key).unwrap_or_else(|_| panic!("missing {key}"));
        t.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }
}

fn models() -> Option<ModelDir> {
    let dir = std::env::var("VOICELAB_MODELS").map(PathBuf::from).unwrap_or_else(|_| repo().join("models"));
    match ModelDir::open(&dir) {
        Ok(m) => Some(m),
        Err(_) => {
            eprintln!("skipping: no models in {} (set VOICELAB_MODELS)", dir.display());
            None
        }
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn rms(a: &[f32]) -> f32 {
    (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt()
}

#[test]
fn mel_banks_match_torchaudio() {
    let f = Fixture::load("dsp.safetensors");
    assert!(max_abs(&dense_banks(), &f.f32("mel_banks")) < 1e-6);
}

#[test]
fn fbank_matches_kaldi() {
    let f = Fixture::load("dsp.safetensors");
    let mut out = Vec::new();
    Fbank::new().compute(&f.f32("fbank_in"), &mut out);
    let err = max_abs(&out, &f.f32("fbank_out"));
    assert!(err < 2e-3, "fbank max abs err {err}");
}

#[test]
fn istft_matches_torch() {
    let f = Fixture::load("dsp.safetensors");
    let y = Istft::new(640, 160).process(&f.f32("istft_real"), &f.f32("istft_imag"), 14);
    let err = max_abs(&y, &f.f32("istft_out"));
    assert!(err < 1e-4, "istft max abs err {err}");
}

fn replay_noise(vc: &mut StreamingVc, noise: Vec<f32>) {
    let mut pos = 0;
    vc.set_noise_source(Box::new(move |buf: &mut [f32]| {
        buf.copy_from_slice(&noise[pos..pos + buf.len()]);
        pos += buf.len();
    }));
}

fn pipeline(variant: Variant, file: &str) {
    let Some(models) = models() else { return };
    let f = Fixture::load(file);
    let mut vc = StreamingVc::new(&models, variant, 1).unwrap();
    vc.set_speaker(&f.f32("spk")).unwrap();
    replay_noise(&mut vc, f.f32("noise"));
    vc.trace = Some(Trace::default());
    let out = vc.convert_clip(&f.f32("input")).unwrap();
    let trace = vc.trace.take().unwrap();

    let bn_err = max_abs(&trace.bn, &f.f32("bn"));
    let mel_err = max_abs(&trace.mel, &f.f32("mel"));
    let want = f.f32("output");
    let diff: Vec<f32> = out.iter().zip(&want).map(|(a, b)| a - b).collect();
    let snr = 20.0 * (rms(&want) / rms(&diff).max(1e-12)).log10();
    eprintln!("{variant:?}: bn err {bn_err:.2e}, mel err {mel_err:.2e}, output SNR {snr:.1} dB");
    assert_eq!(out.len(), want.len());
    assert!(bn_err < 5e-3, "BN features diverge: {bn_err}");
    assert!(mel_err < 2e-2, "mel diverges: {mel_err}");
    assert!(snr > 40.0, "output SNR {snr} dB");
}

#[test]
fn pipeline_quality_matches_reference() {
    pipeline(Variant::Quality, "golden_120ms.safetensors");
}

#[test]
fn pipeline_low_latency_matches_reference() {
    pipeline(Variant::LowLatency, "golden_40ms.safetensors");
}

#[test]
fn pipeline_ultra_matches_reference() {
    pipeline(Variant::Ultra, "golden_ultra.safetensors");
}

/// Streaming bandwidth extension equals the full-utterance reference once past the stream start
/// (zero-padded here, reflected there), whatever the chunk size.
#[test]
fn bwe_streaming_matches_reference() {
    let Some(models) = models() else { return };
    let f = Fixture::load("bwe.safetensors");
    let (x, want) = (f.f32("x48"), f.f32("y48"));
    for chunk in [7680usize, 480, 1000] {
        let mut bwe = Bwe::new(&models, 2).unwrap();
        let mut out = Vec::new();
        for c in x.chunks(chunk) {
            bwe.process(c, &mut out).unwrap();
        }
        bwe.process(&vec![0.0; bwe.latency_samples() + 80], &mut out).unwrap(); // flush
        assert!(out.len() >= want.len(), "chunk {chunk}: {} < {}", out.len(), want.len());
        let skip = 4800; // the first 100 ms see different padding
        // Phase of near-silent bins is numerically arbitrary (f32 here, f64 there), so compare
        // by SNR rather than sample by sample.
        let diff: Vec<f32> = out[skip..want.len()].iter().zip(&want[skip..]).map(|(a, b)| a - b).collect();
        let snr = 20.0 * (rms(&want[skip..]) / rms(&diff).max(1e-12)).log10();
        eprintln!("bwe chunk {chunk}: SNR {snr:.1} dB, max abs err {:.2e}", max_abs(&diff, &vec![0.0; diff.len()]));
        assert!(snr > 30.0, "bwe diverges: SNR {snr} dB");
    }
}

#[test]
fn speaker_embedding_matches_reference() {
    let Some(models) = models() else { return };
    let f = Fixture::load("speaker.safetensors");
    let emb = SpeakerEncoder::new(&models, 4).unwrap().embed(&f.f32("wav")).unwrap();
    let want = f.f32("emb");
    let cos = emb.iter().zip(&want).map(|(a, b)| a * b).sum::<f32>() / (rms(&emb) * rms(&want) * emb.len() as f32);
    assert!(cos > 0.999, "cosine {cos}");
}

/// Feeding the engine its minimum block (80 ms for 40ms, 160 ms for 120ms) or odd sizes must
/// not change the output: the live path relies on this to cut buffering latency.
#[test]
fn block_size_does_not_change_output() {
    let Some(models) = models() else { return };
    for (variant, file) in
        [(Variant::LowLatency, "golden_40ms.safetensors"), (Variant::Quality, "golden_120ms.safetensors")]
    {
        let f = Fixture::load(file);
        let input = f.f32("input");
        let mut outputs = Vec::new();
        for block in [2560usize, 1280, 1000] {
            let mut vc = StreamingVc::new(&models, variant, 1).unwrap();
            vc.set_speaker(&f.f32("spk")).unwrap();
            replay_noise(&mut vc, f.f32("noise"));
            let mut out = Vec::new();
            for chunk in input.chunks(block) {
                out.extend(vc.process(chunk).unwrap());
            }
            outputs.push(out);
        }
        let want = f.f32("output");
        for out in &outputs {
            let n = out.len().min(want.len());
            assert!(n + 2560 >= want.len(), "{variant:?}: output too short ({} vs {})", out.len(), want.len());
            let err = max_abs(&out[..n], &want[..n]);
            assert!(err < 1e-3, "{variant:?}: block size changed output by {err}");
        }
    }
}
