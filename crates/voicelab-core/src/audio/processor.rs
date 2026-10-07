//! Device-independent real-time processing: device-rate mono in -> device-rate mono out for
//! each sink (virtual cable, optional monitor). Owns the model; runs on the worker thread.
//!
//! Chain: [noise suppression at the input rate] -> 16 kHz -> voice conversion (or your own
//! voice) -> gate -> [Ultra: 48 kHz bandwidth extension] -> pitch / effects -> gain -> sinks.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Result;

use crate::dsp::fx::{Denoiser, Effect, FxChain, FxParams};
use crate::dsp::gate::{Gate, peak, rms_db};
use crate::dsp::stream_resample::StreamResampler;
use crate::engine::bwe::{self, Bwe};
use crate::engine::vc::{SAMPLE_RATE, StreamingVc};

/// f32 stored in an `AtomicU32` (lock-free sharing with audio callbacks and the UI).
#[derive(Debug, Default)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }
    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    pub fn store(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
    /// Keep the maximum (for peak meters); returns nothing, readers `take` it.
    pub fn fetch_max(&self, v: f32) {
        let mut cur = self.0.load(Ordering::Relaxed);
        while v > f32::from_bits(cur) {
            match self.0.compare_exchange_weak(cur, v.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
    pub fn take(&self) -> f32 {
        f32::from_bits(self.0.swap(0, Ordering::Relaxed))
    }
}

/// Live-tunable settings, written by the UI and read by the audio threads.
#[derive(Debug)]
pub struct Controls {
    /// Voice conversion on; off = your own voice passes through.
    pub enabled: AtomicBool,
    pub muted: AtomicBool,
    /// Also play the result on the monitor device (your headphones).
    pub monitor: AtomicBool,
    pub input_gain: AtomicF32,
    pub output_gain: AtomicF32,
    /// Gate threshold in dBFS; <= -100 disables the gate.
    pub gate_db: AtomicF32,
    /// RNNoise on the microphone before anything else.
    pub denoise: AtomicBool,
    /// Index into `Effect::ALL`.
    pub effect: AtomicU32,
    /// Pitch shift in semitones ("tono"), on top of the effect's own.
    pub pitch: AtomicF32,
}

impl Controls {
    pub fn set_effect(&self, effect: Effect) {
        let i = Effect::ALL.iter().position(|e| *e == effect).unwrap_or(0);
        self.effect.store(i as u32, Ordering::Relaxed);
    }

    pub fn fx_params(&self) -> FxParams {
        let effect = Effect::ALL.get(self.effect.load(Ordering::Relaxed) as usize).copied().unwrap_or_default();
        FxParams { effect, pitch_semitones: self.pitch.load(), mix: 1.0 }
    }
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            muted: AtomicBool::new(false),
            monitor: AtomicBool::new(false),
            input_gain: AtomicF32::new(1.0),
            output_gain: AtomicF32::new(1.0),
            gate_db: AtomicF32::new(-50.0),
            denoise: AtomicBool::new(false),
            effect: AtomicU32::new(0),
            pitch: AtomicF32::new(0.0),
        }
    }
}

/// Counters for the UI. Peaks are reset when read.
#[derive(Debug, Default)]
pub struct Stats {
    pub input_peak: AtomicF32,
    pub output_peak: AtomicF32,
    pub process_ms: AtomicF32,
    pub process_ms_peak: AtomicF32,
    pub block_ms: AtomicF32,
    pub output_buffer_ms: AtomicF32,
    pub gate_open: AtomicBool,
    /// The input has been exact digital silence for a few seconds (wrong or muted device,
    /// or Windows microphone privacy blocking the app). A real microphone always has noise.
    pub input_silent: AtomicBool,
    pub blocks: AtomicU64,
    pub overruns: AtomicU64,
    pub underruns: AtomicU64,
    /// Glitches reported by the OS audio stack (WASAPI xruns); not fatal.
    pub xruns: AtomicU64,
    pub errors: AtomicU64,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct StatsSnapshot {
    pub input_peak: f32,
    pub output_peak: f32,
    pub process_ms: f32,
    pub process_ms_peak: f32,
    pub block_ms: f32,
    pub load: f32,
    pub output_buffer_ms: f32,
    pub gate_open: bool,
    pub input_silent: bool,
    pub blocks: u64,
    pub overruns: u64,
    pub underruns: u64,
    pub xruns: u64,
    pub errors: u64,
}

impl Stats {
    pub fn snapshot(&self) -> StatsSnapshot {
        let block_ms = self.block_ms.load();
        // Peak over the last second of blocks, maintained by the processor (not reset on read,
        // so the UI can poll faster than blocks arrive).
        let process_ms_peak = self.process_ms_peak.load();
        StatsSnapshot {
            input_peak: self.input_peak.take(),
            output_peak: self.output_peak.take(),
            process_ms: self.process_ms.load(),
            process_ms_peak,
            block_ms,
            load: if block_ms > 0.0 { process_ms_peak / block_ms } else { 0.0 },
            output_buffer_ms: self.output_buffer_ms.load(),
            gate_open: self.gate_open.load(Ordering::Relaxed),
            input_silent: self.input_silent.load(Ordering::Relaxed),
            blocks: self.blocks.load(Ordering::Relaxed),
            overruns: self.overruns.load(Ordering::Relaxed),
            underruns: self.underruns.load(Ordering::Relaxed),
            xruns: self.xruns.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

/// Ultra mode's last stage: 16 kHz -> 48 kHz resampling, then the bandwidth extension.
struct Ultra {
    up: StreamResampler,
    bwe: Bwe,
    x48: Vec<f32>,
}

pub struct Processor {
    vc: StreamingVc,
    block: usize,
    recent_ms: std::collections::VecDeque<f32>,
    silent_samples: usize,
    denoiser: Denoiser,
    denoising: bool,
    denoised: Vec<f32>,
    input: StreamResampler,
    sinks: Vec<StreamResampler>,
    pending: Vec<f32>,
    out16: Vec<f32>,
    ultra: Option<Ultra>,
    /// Final signal at `out_rate` (16 kHz, or 48 kHz in Ultra mode).
    out: Vec<f32>,
    out_rate: u32,
    fx: FxChain,
    gate: Gate,
    was_enabled: bool,
    controls: Arc<Controls>,
    stats: Arc<Stats>,
}

/// Gate hold: longer than the model latency so word endings come through.
const GATE_HOLD_MS: f32 = 450.0;
const GATE_FADE_MS: f32 = 8.0;

impl Processor {
    /// `bwe` turns on Ultra mode's 48 kHz output.
    pub fn new(
        vc: StreamingVc,
        bwe: Option<Bwe>,
        input_rate: u32,
        sink_rates: &[u32],
        controls: Arc<Controls>,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        let block = vc.block_samples();
        stats.block_ms.store(block as f32 * 1000.0 / SAMPLE_RATE as f32);
        let out_rate = if bwe.is_some() { bwe::OUTPUT_RATE } else { SAMPLE_RATE };
        let ultra = match bwe {
            Some(bwe) => Some(Ultra { up: StreamResampler::new(SAMPLE_RATE, out_rate)?, bwe, x48: Vec::new() }),
            None => None,
        };
        Ok(Self {
            block,
            recent_ms: std::collections::VecDeque::new(),
            silent_samples: 0,
            denoiser: Denoiser::new(input_rate),
            denoising: false,
            denoised: Vec::new(),
            input: StreamResampler::new(input_rate, SAMPLE_RATE)?,
            sinks: sink_rates.iter().map(|&r| StreamResampler::new(out_rate, r)).collect::<Result<_>>()?,
            pending: Vec::new(),
            out16: Vec::new(),
            ultra,
            out: Vec::new(),
            out_rate,
            fx: FxChain::new(out_rate),
            gate: Gate::new(SAMPLE_RATE, GATE_HOLD_MS, GATE_FADE_MS),
            was_enabled: true,
            vc,
            controls,
            stats,
        })
    }

    pub fn vc_mut(&mut self) -> &mut StreamingVc {
        &mut self.vc
    }

    /// Input block size in samples at 16 kHz.
    pub fn block(&self) -> usize {
        self.block
    }

    /// Rate of the processed signal before the per-sink resamplers.
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    /// Delay added after the model: resampling to 48 kHz and the bandwidth extension (Ultra).
    pub fn post_latency_ms(&self) -> f32 {
        self.ultra
            .as_ref()
            .map_or(0.0, |u| (u.up.delay() + u.bwe.latency_samples()) as f32 * 1000.0 / bwe::OUTPUT_RATE as f32)
    }

    /// Delay of the noise suppressor, when on.
    pub fn denoise_latency_ms(&self) -> f32 {
        self.denoiser.latency_samples() as f32 * 1000.0 / self.denoiser.sample_rate() as f32
    }

    /// Feed device-rate input; appends device-rate output to `outs[i]` for each sink.
    pub fn push(&mut self, input: &[f32], outs: &mut [Vec<f32>]) -> Result<()> {
        let denoise = self.controls.denoise.load(Ordering::Relaxed);
        if denoise && !self.denoising {
            self.denoiser.reset();
        }
        self.denoising = denoise;
        if denoise {
            self.denoised.clear();
            self.denoiser.process(input, &mut self.denoised);
            self.input.push(&self.denoised, &mut self.pending);
        } else {
            self.input.push(input, &mut self.pending);
        }
        while self.pending.len() >= self.block {
            let block: Vec<f32> = self.pending.drain(..self.block).collect();
            self.process_block(&block, outs)?;
        }
        Ok(())
    }

    fn process_block(&mut self, block: &[f32], outs: &mut [Vec<f32>]) -> Result<()> {
        let c = &self.controls;
        let enabled = c.enabled.load(Ordering::Relaxed);
        let t0 = Instant::now();
        self.out16.clear();
        if enabled {
            if !self.was_enabled {
                self.vc.reset();
            }
            self.out16.extend(self.vc.process(block)?);
        } else {
            self.out16.extend_from_slice(block);
        }
        self.was_enabled = enabled;
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.stats.process_ms.store(ms);
        let window = (SAMPLE_RATE as usize / self.block).max(1); // ~1 s of blocks
        self.recent_ms.push_back(ms);
        while self.recent_ms.len() > window {
            self.recent_ms.pop_front();
        }
        self.stats.process_ms_peak.store(self.recent_ms.iter().copied().fold(0.0, f32::max));
        self.stats.blocks.fetch_add(1, Ordering::Relaxed);

        self.silent_samples = if block.iter().all(|&s| s == 0.0) { self.silent_samples + block.len() } else { 0 };
        self.stats.input_silent.store(self.silent_samples >= 3 * SAMPLE_RATE as usize, Ordering::Relaxed);

        let gain = if c.muted.load(Ordering::Relaxed) { 0.0 } else { c.output_gain.load() };
        self.gate.process(rms_db(block), c.gate_db.load(), &mut self.out16);
        self.out.clear();
        match &mut self.ultra {
            Some(u) => {
                u.x48.clear();
                u.up.push(&self.out16, &mut u.x48);
                u.bwe.process(&u.x48, &mut self.out)?;
            }
            None => self.out.extend_from_slice(&self.out16),
        }
        self.fx.set_params(c.fx_params());
        self.fx.process(&mut self.out);
        for s in &mut self.out {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
        self.stats.gate_open.store(self.gate.is_open(), Ordering::Relaxed);
        self.stats.output_peak.fetch_max(peak(&self.out));
        for (sink, out) in self.sinks.iter_mut().zip(outs.iter_mut()) {
            sink.push(&self.out, out);
        }
        Ok(())
    }
}
