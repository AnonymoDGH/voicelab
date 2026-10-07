//! Streaming voice conversion: 16 kHz mono in, 16 kHz mono out.
//!
//! Port of `tools/export/voicelab_export/onnx_pipeline.py::OnnxVC` (itself a port of MeanVC2's
//! `runtime/run_rt.py`). Pipeline per 160 ms block:
//! fbank -> Fast-U2++ BN features (sliding window, cached attention/conv state) ->
//! linear upsampling to 16 frames -> DiT mean-flow (2 ODE steps, KV cache) -> Vocos -> iSTFT ->
//! overlap-add crossfade.

use anyhow::{Context, Result, ensure};
use ort::session::Session;
use ort::value::TensorRef;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

use super::models::{ModelDir, Variant, VariantSpec};
use crate::dsp::fbank::{FRAME_LEN, FRAME_SHIFT, Fbank, N_MELS};
use crate::dsp::interp_linear;
use crate::dsp::istft::Istft;

pub const SAMPLE_RATE: u32 = 16000;
/// Samples per `process` call in the reference implementation (160 ms).
pub const BLOCK: usize = 2560;
/// Each BN frame covers 40 ms; the DiT works on 10 ms mel frames.
const UPSAMPLE_BN: usize = 4;
pub const SPK_DIM: usize = 256;
const BN_DIM: usize = 256;
const VOCODER_OVERLAP: usize = 2;
const UPSAMPLE: usize = 160;
const MEM_SLOTS: usize = 32;
const OFFSET_WRAP: i64 = 4000;

/// Fills a buffer with the ODE start noise (standard normal).
pub type NoiseSource = Box<dyn FnMut(&mut [f32]) + Send>;

pub fn gaussian_noise(seed: u64) -> NoiseSource {
    let mut rng = SmallRng::seed_from_u64(seed);
    Box::new(move |buf: &mut [f32]| {
        for v in buf {
            *v = StandardNormal.sample(&mut rng);
        }
    })
}

/// Per-stage intermediates, recorded only when `StreamingVc::trace` is set (tests).
#[doc(hidden)]
#[derive(Default)]
pub struct Trace {
    pub bn: Vec<f32>,
    pub mel: Vec<f32>,
}

pub struct StreamingVc {
    #[doc(hidden)]
    pub trace: Option<Trace>,
    variant: Variant,
    spec: VariantSpec,
    asr: Session,
    dit: Session,
    gtm: Session,
    vocos: Session,
    fbank: Fbank,
    istft: Istft,
    noise: NoiseSource,
    ramp_down: Vec<f32>,
    ramp_up: Vec<f32>,

    spk: Vec<f32>,
    k_mem: Vec<f32>,
    v_mem: Vec<f32>,

    samples_cache: Vec<f32>,
    frame_cache: Vec<f32>,
    enc_cache: Option<Vec<f32>>,
    asr_offset: i64,
    att_cache: Vec<f32>,
    cnn_cache: Vec<f32>,
    kv: Vec<f32>,
    kv_valid: usize,
    vc_offset: i64,
    noise_cache: Option<Vec<f32>>,
    vocoder_cache: Option<Vec<f32>>,
    last_wav: Option<Vec<f32>>,
    bn_buffer: Vec<f32>,
}

pub(crate) fn run_err(e: ort::Error) -> anyhow::Error {
    anyhow::anyhow!("ONNX Runtime: {e}")
}

impl StreamingVc {
    pub fn new(models: &ModelDir, variant: Variant, threads: usize) -> Result<Self> {
        let spec = models.variant(variant)?.clone();
        let voc = &models.manifest.vocoder;
        let ramp = |a: f32, b: f32| (0..UPSAMPLE).map(|i| a + (b - a) * i as f32 / (UPSAMPLE - 1) as f32).collect();
        let mut vc = Self {
            trace: None,
            variant,
            asr: models.session(&spec.asr, threads)?,
            dit: models.session(&spec.dit, threads)?,
            gtm: models.session(&spec.gtm, 1)?,
            vocos: models.session(&voc.file, threads)?,
            fbank: Fbank::new(),
            istft: Istft::new(voc.n_fft, voc.hop),
            noise: gaussian_noise(0),
            ramp_down: ramp(1.0, 0.0),
            ramp_up: ramp(0.0, 1.0),
            spk: vec![0.0; SPK_DIM],
            k_mem: Vec::new(),
            v_mem: Vec::new(),
            samples_cache: Vec::new(),
            frame_cache: Vec::new(),
            enc_cache: None,
            asr_offset: 0,
            att_cache: Vec::new(),
            cnn_cache: Vec::new(),
            kv: Vec::new(),
            kv_valid: 0,
            vc_offset: 0,
            noise_cache: None,
            vocoder_cache: None,
            last_wav: None,
            bn_buffer: Vec::new(),
            spec,
        };
        ensure!(voc.hop == UPSAMPLE, "vocoder hop {} != {UPSAMPLE}", voc.hop);
        vc.set_speaker(&[0.0; SPK_DIM])?;
        vc.reset();
        Ok(vc)
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// Smallest useful input block: one ASR stride (160 ms for 120ms, 80 ms for 40ms).
    /// Feeding blocks of this size gives the same output as the 160 ms reference cadence,
    /// with less buffering delay.
    pub fn block_samples(&self) -> usize {
        self.spec.bn_stride * FRAME_SHIFT
    }

    /// Algorithmic delay in samples: one input block, ASR lookahead, DiT lookahead and the
    /// vocoder overlap. Excludes audio-device buffering and compute time.
    pub fn algorithmic_latency_samples(&self) -> usize {
        let s = &self.spec;
        let asr_lookahead = (s.bn_window - s.bn_stride) * FRAME_SHIFT + (FRAME_LEN - FRAME_SHIFT);
        self.block_samples() + asr_lookahead + (s.block_size + VOCODER_OVERLAP) * UPSAMPLE
    }

    pub fn set_noise_source(&mut self, noise: NoiseSource) {
        self.noise = noise;
    }

    /// Switch the target voice. Takes effect on the next window; streaming state is kept,
    /// so voices can be changed while talking.
    pub fn set_speaker(&mut self, emb: &[f32]) -> Result<()> {
        ensure!(emb.len() == SPK_DIM, "speaker embedding must have {SPK_DIM} values");
        let out = self
            .gtm
            .run(ort::inputs!["spk" => TensorRef::from_array_view(([1usize, SPK_DIM], emb)).map_err(run_err)?])
            .map_err(run_err)?;
        self.k_mem = out["k_mem"].try_extract_tensor::<f32>().map_err(run_err)?.1.to_vec();
        self.v_mem = out["v_mem"].try_extract_tensor::<f32>().map_err(run_err)?.1.to_vec();
        drop(out);
        self.spk = emb.to_vec();
        Ok(())
    }

    /// Clear all streaming state (call when the input stream restarts).
    pub fn reset(&mut self) {
        let s = &self.spec;
        self.samples_cache.clear();
        self.frame_cache.clear();
        self.enc_cache = None;
        self.asr_offset = s.asr_offset_init;
        self.att_cache = vec![0.0; 6 * 4 * s.asr_cache * 128];
        self.cnn_cache = vec![0.0; 6 * 256 * 8];
        self.kv = vec![0.0; s.kv_shape.iter().product()];
        self.kv_valid = 0;
        self.vc_offset = 0;
        self.noise_cache = None;
        self.vocoder_cache = None;
        self.last_wav = None;
        self.bn_buffer.clear();
    }

    /// Feed 16 kHz mono samples (any length; 2560 = 160 ms matches the reference cadence).
    /// Returns the converted audio produced so far (possibly empty).
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        let Some(bn) = self.encode(samples)? else { return Ok(out) };
        self.bn_buffer.extend_from_slice(&bn);
        let window = self.spec.chunk_size + self.spec.block_size;
        while self.bn_buffer.len() >= window * BN_DIM {
            let cond = self.bn_buffer[..window * BN_DIM].to_vec();
            self.bn_buffer.drain(..self.spec.chunk_size * BN_DIM);
            let mel = self.vc_step(&cond)?;
            out.extend(self.decode(&mel)?);
        }
        Ok(out)
    }

    fn encode(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        let mut padded = std::mem::take(&mut self.samples_cache);
        padded.extend_from_slice(samples);
        if padded.len() < FRAME_LEN {
            self.samples_cache = padded;
            return Ok(None);
        }
        let mut fb = std::mem::take(&mut self.frame_cache);
        let n = self.fbank.compute(&padded, &mut fb);
        self.samples_cache = padded[FRAME_SHIFT * n..].to_vec();

        let s = &self.spec;
        if self.asr_offset >= OFFSET_WRAP {
            self.asr_offset = (s.asr_cache as i64).max(self.asr_offset - OFFSET_WRAP);
        }
        let frames = fb.len() / N_MELS;
        let mut bn = Vec::new();
        let mut i = 0;
        while i + s.bn_window <= frames {
            let window = &fb[i * N_MELS..(i + s.bn_window) * N_MELS];
            let offset = [self.asr_offset];
            let rcs = [s.asr_cache as i64];
            let out = self
                .asr
                .run(ort::inputs![
                    "fbank" => TensorRef::from_array_view(([1usize, s.bn_window, N_MELS], window)).map_err(run_err)?,
                    "offset" => TensorRef::from_array_view(((), &offset[..])).map_err(run_err)?,
                    "required_cache_size" => TensorRef::from_array_view(((), &rcs[..])).map_err(run_err)?,
                    "att_cache" => TensorRef::from_array_view(([6usize, 4, s.asr_cache, 128], &self.att_cache[..])).map_err(run_err)?,
                    "cnn_cache" => TensorRef::from_array_view(([6usize, 1, 256, 8], &self.cnn_cache[..])).map_err(run_err)?,
                ])
                .map_err(run_err)?;
            let new_bn = out["bn"].try_extract_tensor::<f32>().map_err(run_err)?.1;
            if let Some(t) = &mut self.trace {
                t.bn.extend_from_slice(new_bn);
            }
            bn.extend_from_slice(new_bn);
            self.att_cache.copy_from_slice(out["att_cache_out"].try_extract_tensor::<f32>().map_err(run_err)?.1);
            self.cnn_cache.copy_from_slice(out["cnn_cache_out"].try_extract_tensor::<f32>().map_err(run_err)?.1);
            drop(out);
            self.asr_offset += s.asr_offset_step;
            i += s.bn_stride;
        }
        self.frame_cache = fb[i * N_MELS..].to_vec();

        if bn.is_empty() {
            return Ok(None);
        }
        // Prepend the last frame of the previous call; at stream start replicate the first
        // frame instead (MeanVC2 stretched the first block's frames over 160 ms instead).
        let mut all = self.enc_cache.take().unwrap_or_else(|| bn[..BN_DIM].to_vec());
        all.extend_from_slice(&bn);
        let bn = all;
        let len = bn.len() / BN_DIM;
        self.enc_cache = Some(bn[(len - 1) * BN_DIM..].to_vec());
        // align_corners interpolation to 4(len-1)+1 points == exact 4x upsampling of each
        // BN interval, so the result does not depend on how input is split into blocks.
        let up = interp_linear(&bn, BN_DIM, UPSAMPLE_BN * (len - 1) + 1);
        Ok(Some(up[BN_DIM..].to_vec()))
    }

    fn vc_step(&mut self, cond: &[f32]) -> Result<Vec<f32>> {
        let s = &self.spec;
        let window = s.chunk_size + s.block_size;
        let mut x = vec![0.0f32; window * N_MELS];
        (self.noise)(&mut x);
        if let Some(cache) = &self.noise_cache {
            x[..s.block_size * N_MELS].copy_from_slice(cache);
        }
        self.noise_cache = Some(x[(window - s.block_size) * N_MELS..].to_vec());
        if self.vc_offset >= OFFSET_WRAP {
            self.vc_offset = 0;
            self.kv.fill(0.0);
            self.kv_valid = 0;
        }
        let valid = [self.kv_valid as i64];
        let out = self
            .dit
            .run(ort::inputs![
                "x" => TensorRef::from_array_view(([1usize, window, N_MELS], &x[..])).map_err(run_err)?,
                "cond" => TensorRef::from_array_view(([1usize, window, BN_DIM], cond)).map_err(run_err)?,
                "spk" => TensorRef::from_array_view(([1usize, SPK_DIM], &self.spk[..])).map_err(run_err)?,
                "k_mem" => TensorRef::from_array_view(([1usize, MEM_SLOTS, BN_DIM], &self.k_mem[..])).map_err(run_err)?,
                "v_mem" => TensorRef::from_array_view(([1usize, MEM_SLOTS, BN_DIM], &self.v_mem[..])).map_err(run_err)?,
                "kv" => TensorRef::from_array_view((s.kv_shape.clone(), &self.kv[..])).map_err(run_err)?,
                "kv_valid" => TensorRef::from_array_view(((), &valid[..])).map_err(run_err)?,
            ])
            .map_err(run_err)?;
        let mel = out["mel"].try_extract_tensor::<f32>().map_err(run_err)?.1.to_vec();
        self.kv.copy_from_slice(out["kv_out"].try_extract_tensor::<f32>().map_err(run_err)?.1);
        drop(out);
        self.kv_valid = (self.kv_valid + s.chunk_size).min(s.cache_frames);
        self.vc_offset += s.chunk_size as i64;
        if let Some(t) = &mut self.trace {
            t.mel.extend_from_slice(&mel);
        }
        Ok(mel) // [chunk, 80]
    }

    fn decode(&mut self, mel: &[f32]) -> Result<Vec<f32>> {
        let mut frames = self.vocoder_cache.take().unwrap_or_default();
        frames.extend_from_slice(mel);
        let t = frames.len() / N_MELS;
        self.vocoder_cache = Some(frames[(t - VOCODER_OVERLAP) * N_MELS..].to_vec());
        // [T, 80] model range -> [1, 80, T] vocoder range
        let mut voc_in = vec![0.0f32; N_MELS * t];
        for f in 0..t {
            for m in 0..N_MELS {
                voc_in[m * t + f] = (frames[f * N_MELS + m] + 1.0) * 0.5;
            }
        }
        let out = self
            .vocos
            .run(
                ort::inputs!["mel" => TensorRef::from_array_view(([1usize, N_MELS, t], &voc_in[..])).map_err(run_err)?],
            )
            .map_err(run_err)?;
        let real = out["real"].try_extract_tensor::<f32>().map_err(run_err)?.1;
        let imag = out["imag"].try_extract_tensor::<f32>().map_err(run_err)?.1;
        let wav = self.istft.process(real, imag, t);
        drop(out);

        let n = wav.len();
        let result = match &self.last_wav {
            Some(last) => {
                let mut r = Vec::with_capacity(n - UPSAMPLE);
                r.extend((0..UPSAMPLE).map(|i| last[i] * self.ramp_down[i] + wav[i] * self.ramp_up[i]));
                r.extend_from_slice(&wav[UPSAMPLE..n - UPSAMPLE]);
                r
            }
            None => wav[..n - UPSAMPLE].to_vec(),
        };
        self.last_wav = Some(wav[n - UPSAMPLE..].to_vec());
        Ok(result)
    }

    /// Convert a whole 16 kHz clip, fed in `BLOCK`-sized pieces like the live path.
    pub fn convert_clip(&mut self, wav: &[f32]) -> Result<Vec<f32>> {
        self.reset();
        let mut out = Vec::with_capacity(wav.len());
        for block in wav.chunks(BLOCK) {
            out.extend(self.process(block).context("converting block")?);
        }
        Ok(out)
    }
}
