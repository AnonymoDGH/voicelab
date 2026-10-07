//! Live engine: microphone -> voice conversion -> virtual cable (+ optional monitor).
//!
//! Threads:
//! * cpal input callback: downmix, gain, push into a lock-free ring (no allocation, no locks).
//! * worker: pops input, resamples to 16 kHz, runs the model per block, resamples per sink and
//!   pushes into one ring per output.
//! * cpal output callbacks: pop from their ring. A ring starts playing once it holds one block
//!   plus a safety margin; if it grows past that (the two devices' clocks drift) the excess is
//!   dropped, and an underrun re-primes it.
//! * host thread: owns the cpal streams (not `Send` on every platform) until `stop`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, StreamConfig};
use rtrb::{Consumer, Producer, RingBuffer};

use super::devices::{self, Direction};
use super::processor::{Controls, Processor, Stats, StatsSnapshot};
use crate::engine::vc::{SAMPLE_RATE, StreamingVc};
use crate::engine::{Bwe, ModelDir, Variant};

/// Extra output buffering on top of one block, to absorb processing jitter. Ultra's longer
/// compute varies more. The output trims what it does not need during silences.
const SAFETY_MARGIN_MS: u32 = 40;
const SAFETY_MARGIN_ULTRA_MS: u32 = 60;
/// Smallest cushion the adaptive trim keeps; grows after each underrun.
const MIN_CUSHION_MS: u32 = 12;
/// How long the buffer level is observed before trimming.
const TRIM_WINDOW_S: u32 = 3;
const RING_SECONDS: u32 = 2;

#[derive(Debug, Clone, Default)]
pub struct LiveConfig {
    /// Input device id or name fragment; `None` = system default microphone.
    pub input: Option<String>,
    /// Output device; `None` = VB-Cable if installed, else the default output.
    pub output: Option<String>,
    /// Monitor (headphones) device; `None` = no monitor stream.
    pub monitor: Option<String>,
    pub variant: Option<Variant>,
    pub threads: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveInfo {
    pub input: String,
    pub output: String,
    pub monitor: Option<String>,
    pub output_is_virtual_cable: bool,
    pub input_rate: u32,
    pub output_rate: u32,
    pub block_ms: f32,
    /// Model + buffering latency estimate (excludes the devices' own buffers).
    pub latency_ms: f32,
}

enum Command {
    SetVoice(Vec<f32>),
    Reset,
}

pub struct LiveEngine {
    info: LiveInfo,
    controls: Arc<Controls>,
    stats: Arc<Stats>,
    commands: mpsc::Sender<Command>,
    stop: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,
    threads: Vec<JoinHandle<()>>,
}

struct HostReady {
    input_rate: u32,
    sink_rates: Vec<u32>,
    input: Consumer<f32>,
    sinks: Vec<Producer<f32>>,
    input_name: String,
    output_name: String,
    monitor_name: Option<String>,
}

impl LiveEngine {
    pub fn start(models: &ModelDir, cfg: LiveConfig, voice: &[f32], controls: Arc<Controls>) -> Result<Self> {
        let variant = cfg.variant.unwrap_or(Variant::LowLatency);
        // Ultra needs the parallelism: 4 flow steps plus the 48 kHz extension per block.
        let threads = if variant.uses_bwe() { cfg.threads.max(2) } else { cfg.threads.max(1) };
        let mut vc = StreamingVc::new(models, variant, threads)?;
        vc.set_speaker(voice)?;
        let bwe = if variant.uses_bwe() { Some(Bwe::new(models, threads)?) } else { None };
        let margin_ms = if variant.uses_bwe() { SAFETY_MARGIN_ULTRA_MS } else { SAFETY_MARGIN_MS };
        let block = vc.block_samples();
        let model_latency = vc.algorithmic_latency_samples();

        let stats = Arc::new(Stats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let last_error = Arc::new(Mutex::new(None));
        let output = cfg.output.clone().or_else(|| devices::find_virtual_cable().map(|d| d.id));
        let output_is_virtual_cable = devices::list(Direction::Output)
            .iter()
            .any(|d| d.is_virtual_cable && output.as_deref().is_some_and(|o| o == d.id || d.name.contains(o)));

        let (ready_tx, ready_rx) = mpsc::channel();
        let host = {
            let (stop, stats, controls, last_error) =
                (stop.clone(), stats.clone(), controls.clone(), last_error.clone());
            let (input, monitor) = (cfg.input.clone(), cfg.monitor.clone());
            std::thread::Builder::new().name("voicelab-audio-host".into()).spawn(move || {
                let streams =
                    match open_streams(input, output, monitor, (block, margin_ms), &controls, &stats, &last_error) {
                        Ok((streams, ready)) => {
                            let _ = ready_tx.send(Ok(ready));
                            streams
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                for s in &streams {
                    if let Err(e) = s.play() {
                        *last_error.lock().unwrap() = Some(format!("no se pudo iniciar el audio: {e}"));
                    }
                }
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50));
                }
                drop(streams);
            })?
        };
        let ready = ready_rx.recv().map_err(|_| anyhow!("el hilo de audio terminó inesperadamente"))??;

        let processor = Processor::new(vc, bwe, ready.input_rate, &ready.sink_rates, controls.clone(), stats.clone())?;
        let post_latency_ms = processor.post_latency_ms();
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let worker = {
            let (stop, stats, controls, last_error) =
                (stop.clone(), stats.clone(), controls.clone(), last_error.clone());
            let HostReady { input, sinks, .. } = ready;
            std::thread::Builder::new().name("voicelab-worker".into()).spawn(move || {
                if let Err(e) = worker_loop(processor, input, sinks, cmd_rx, &stop, &stats, &controls) {
                    *last_error.lock().unwrap() = Some(format!("{e:#}"));
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                }
            })?
        };

        let ms = |s: usize| s as f32 * 1000.0 / SAMPLE_RATE as f32;
        let info = LiveInfo {
            input: ready.input_name,
            output: ready.output_name,
            monitor: ready.monitor_name,
            output_is_virtual_cable,
            input_rate: ready.input_rate,
            output_rate: ready.sink_rates[0],
            block_ms: ms(block),
            latency_ms: ms(model_latency) + post_latency_ms + margin_ms as f32,
        };
        Ok(Self { info, controls, stats, commands: cmd_tx, stop, last_error, threads: vec![host, worker] })
    }

    pub fn info(&self) -> &LiveInfo {
        &self.info
    }

    pub fn controls(&self) -> &Arc<Controls> {
        &self.controls
    }

    pub fn stats(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    /// Error that stopped or disturbed the engine (device unplugged, model failure).
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().unwrap().clone()
    }

    pub fn is_running(&self) -> bool {
        self.threads.iter().all(|t| !t.is_finished())
    }

    pub fn set_voice(&self, embedding: Vec<f32>) {
        let _ = self.commands.send(Command::SetVoice(embedding));
    }

    pub fn reset(&self) {
        let _ = self.commands.send(Command::Reset);
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for LiveEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(
    mut processor: Processor,
    mut input: Consumer<f32>,
    mut sinks: Vec<Producer<f32>>,
    commands: mpsc::Receiver<Command>,
    stop: &AtomicBool,
    stats: &Stats,
    controls: &Controls,
) -> Result<()> {
    let mut buf = Vec::with_capacity(SAMPLE_RATE as usize);
    let mut outs = vec![Vec::new(); sinks.len()];
    while !stop.load(Ordering::Relaxed) {
        while let Ok(cmd) = commands.try_recv() {
            match cmd {
                Command::SetVoice(emb) => processor.vc_mut().set_speaker(&emb)?,
                Command::Reset => processor.vc_mut().reset(),
            }
        }
        let n = input.slots();
        if n == 0 {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        let chunk = input.read_chunk(n).expect("slots checked");
        let (a, b) = chunk.as_slices();
        buf.clear();
        buf.extend_from_slice(a);
        buf.extend_from_slice(b);
        chunk.commit_all();

        processor.push(&buf, &mut outs)?;
        let monitor_on = controls.monitor.load(Ordering::Relaxed);
        for (i, (out, sink)) in outs.iter_mut().zip(&mut sinks).enumerate() {
            if i == 0 || monitor_on {
                let n = out.len().min(sink.slots());
                if n < out.len() {
                    stats.overruns.fetch_add(1, Ordering::Relaxed);
                }
                if let Ok(mut w) = sink.write_chunk_uninit(n) {
                    let (a, b) = w.as_mut_slices();
                    for (d, s) in a.iter_mut().chain(b.iter_mut()).zip(out.iter()) {
                        d.write(*s);
                    }
                    unsafe { w.commit_all() };
                }
            }
            out.clear();
        }
    }
    Ok(())
}

type Streams = Vec<cpal::Stream>;

fn open_streams(
    input: Option<String>,
    output: Option<String>,
    monitor: Option<String>,
    (block16k, margin_ms): (usize, u32),
    controls: &Arc<Controls>,
    stats: &Arc<Stats>,
    last_error: &Arc<Mutex<Option<String>>>,
) -> Result<(Streams, HostReady)> {
    let (in_dev, input_name) = devices::open(Direction::Input, input.as_deref())?;
    let in_cfg = in_dev.default_input_config().context("el micrófono no informa su formato")?;
    let input_rate = in_cfg.sample_rate();
    let (in_prod, in_cons) = RingBuffer::new((input_rate * RING_SECONDS) as usize);
    let mut streams =
        vec![build_input(&in_dev, in_cfg.sample_format(), in_cfg.into(), in_prod, controls, stats, last_error)?];

    let mut sink_rates = Vec::new();
    let mut sinks = Vec::new();
    let mut names = Vec::new();
    for (i, wanted) in [Some(output), monitor.map(Some)].into_iter().flatten().enumerate() {
        let (dev, name) = devices::open(Direction::Output, wanted.as_deref())?;
        let cfg = dev.default_output_config().with_context(|| format!("«{name}» no informa su formato"))?;
        let rate = cfg.sample_rate();
        let (prod, cons) = RingBuffer::new((rate * RING_SECONDS) as usize);
        let block = (block16k as u64 * rate as u64 / SAMPLE_RATE as u64) as usize;
        let start = block + (rate * margin_ms / 1000) as usize;
        let state = OutputState::new(cons, start, block, cfg.channels() as usize, rate, i == 0, stats.clone());
        streams.push(build_output(&dev, cfg.sample_format(), cfg.into(), state, stats, last_error)?);
        sink_rates.push(rate);
        sinks.push(prod);
        names.push(name);
    }
    let mut names = names.into_iter();
    let ready = HostReady {
        input_rate,
        sink_rates,
        input: in_cons,
        sinks,
        input_name,
        output_name: names.next().expect("main output"),
        monitor_name: names.next(),
    };
    Ok((streams, ready))
}

fn error_callback(
    stats: &Arc<Stats>,
    last_error: &Arc<Mutex<Option<String>>>,
) -> impl FnMut(cpal::Error) + Send + 'static {
    let (stats, last_error) = (stats.clone(), last_error.clone());
    move |e| match e.kind() {
        // A glitch the OS already recovered from, or the default device being rerouted.
        cpal::ErrorKind::Xrun => {
            stats.xruns.fetch_add(1, Ordering::Relaxed);
        }
        cpal::ErrorKind::DeviceChanged => {}
        _ => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut g) = last_error.lock() {
                *g = Some(format!("error de audio: {e}"));
            }
        }
    }
}

fn build_input(
    dev: &cpal::Device,
    format: SampleFormat,
    cfg: StreamConfig,
    prod: Producer<f32>,
    controls: &Arc<Controls>,
    stats: &Arc<Stats>,
    last_error: &Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream> {
    fn build<T: SizedSample>(
        dev: &cpal::Device,
        cfg: StreamConfig,
        mut prod: Producer<f32>,
        controls: Arc<Controls>,
        stats: Arc<Stats>,
        err: impl FnMut(cpal::Error) + Send + 'static,
    ) -> Result<cpal::Stream>
    where
        f32: FromSample<T>,
    {
        let ch = cfg.channels as usize;
        let stream = dev.build_input_stream(
            cfg,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let gain = controls.input_gain.load();
                let mut peak = 0.0f32;
                for frame in data.chunks_exact(ch) {
                    let v = frame.iter().map(|s| f32::from_sample(*s)).sum::<f32>() / ch as f32 * gain;
                    peak = peak.max(v.abs());
                    if prod.push(v).is_err() {
                        stats.overruns.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
                stats.input_peak.fetch_max(peak);
            },
            err,
            None,
        )?;
        Ok(stream)
    }
    let (c, s, err) = (controls.clone(), stats.clone(), error_callback(stats, last_error));
    match format {
        SampleFormat::F32 => build::<f32>(dev, cfg, prod, c, s, err),
        SampleFormat::I16 => build::<i16>(dev, cfg, prod, c, s, err),
        SampleFormat::I32 => build::<i32>(dev, cfg, prod, c, s, err),
        SampleFormat::U16 => build::<u16>(dev, cfg, prod, c, s, err),
        other => bail!("formato de micrófono no soportado: {other}"),
    }
}

struct OutputState {
    ring: Consumer<f32>,
    priming: bool,
    start: usize,
    high: usize,
    channels: usize,
    rate: u32,
    main: bool,
    stats: Arc<Stats>,
    /// Adaptive latency: the lowest buffer level seen in the current window, the cushion to
    /// keep, and how many samples are still to be dropped. Only silent samples are dropped, so
    /// trimming is inaudible; it happens between phrases.
    floor: usize,
    window_left: usize,
    cushion: usize,
    trim: usize,
}

/// Samples quieter than this count as silence (the gate outputs exact zeros).
const SILENCE: f32 = 1e-4;

impl OutputState {
    fn new(
        ring: Consumer<f32>,
        start: usize,
        block: usize,
        channels: usize,
        rate: u32,
        main: bool,
        stats: Arc<Stats>,
    ) -> Self {
        Self {
            ring,
            priming: true,
            start,
            high: start + block + (rate / 20) as usize,
            channels,
            rate,
            main,
            stats,
            floor: usize::MAX,
            window_left: (rate * TRIM_WINDOW_S) as usize,
            cushion: (rate * MIN_CUSHION_MS / 1000) as usize,
            trim: 0,
        }
    }

    fn fill<T: Sample + FromSample<f32>>(&mut self, data: &mut [T]) {
        let frames = data.len() / self.channels;
        let mut avail = self.ring.slots();
        if self.priming {
            if avail < self.start {
                data.fill(T::EQUILIBRIUM);
                return;
            }
            self.priming = false;
            self.floor = usize::MAX;
            self.window_left = (self.rate * TRIM_WINDOW_S) as usize;
        }
        if avail > self.high {
            // Input clock faster than output clock: drop the excess to keep latency bounded.
            let excess = avail - self.start;
            if let Ok(c) = self.ring.read_chunk(excess) {
                c.commit_all();
            }
            avail = self.ring.slots();
        }
        // Read up to one extra callback's worth when trimming; silent samples are skipped as
        // long as enough remain to fill this callback.
        let take = (frames + self.trim.min(frames)).min(avail);
        let mut used = 0;
        let mut written = 0;
        if let Ok(chunk) = self.ring.read_chunk(take) {
            let (a, b) = chunk.as_slices();
            let mut samples = a.iter().chain(b);
            for frame in data.chunks_exact_mut(self.channels) {
                let Some(mut v) = samples.next().copied() else { break };
                used += 1;
                while self.trim > 0 && v.abs() < SILENCE && take - used >= frames - written {
                    let Some(next) = samples.next().copied() else { break };
                    used += 1;
                    self.trim -= 1;
                    v = next;
                }
                frame.fill(T::from_sample(v));
                written += 1;
            }
            chunk.commit(used);
        }
        if written < frames {
            data[written * self.channels..].fill(T::EQUILIBRIUM);
            self.priming = true;
            // The cushion was too thin for this machine: keep more from now on.
            self.cushion = (self.cushion + (self.rate / 100) as usize).min((self.rate / 10) as usize);
            self.trim = 0;
            if self.main {
                self.stats.underruns.fetch_add(1, Ordering::Relaxed);
            }
        }
        let level = avail - used;
        self.floor = self.floor.min(level);
        self.window_left = self.window_left.saturating_sub(frames);
        if self.window_left == 0 {
            // Never needed more than `cushion` of what was buffered: drop the rest when quiet.
            self.trim = self.floor.saturating_sub(self.cushion);
            self.floor = usize::MAX;
            self.window_left = (self.rate * TRIM_WINDOW_S) as usize;
        }
        if self.main {
            self.stats.output_buffer_ms.store(level as f32 * 1000.0 / self.rate as f32);
        }
    }
}

fn build_output(
    dev: &cpal::Device,
    format: SampleFormat,
    cfg: StreamConfig,
    state: OutputState,
    stats: &Arc<Stats>,
    last_error: &Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream> {
    fn build<T: SizedSample + FromSample<f32>>(
        dev: &cpal::Device,
        cfg: StreamConfig,
        mut state: OutputState,
        err: impl FnMut(cpal::Error) + Send + 'static,
    ) -> Result<cpal::Stream> {
        Ok(dev.build_output_stream(
            cfg,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| state.fill(data),
            err,
            None,
        )?)
    }
    let err = error_callback(stats, last_error);
    match format {
        SampleFormat::F32 => build::<f32>(dev, cfg, state, err),
        SampleFormat::I16 => build::<i16>(dev, cfg, state, err),
        SampleFormat::I32 => build::<i32>(dev, cfg, state, err),
        SampleFormat::U16 => build::<u16>(dev, cfg, state, err),
        other => bail!("formato de salida no soportado: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulates 10 minutes in 10 ms output-clock ticks. The input clock runs `drift` faster
    /// (positive) or slower (negative) than the output clock; the worker emits one 80 ms block
    /// whenever enough input has arrived. Returns (underruns after priming, max ring level).
    fn simulate(drift: f64) -> (u64, usize) {
        let (underruns, max_level, _) = simulate_with(drift, 0.5);
        (underruns, max_level)
    }

    /// Like `simulate` with a constant signal `value`; also returns the lowest buffer level over
    /// the last 5 seconds.
    fn simulate_with(drift: f64, value: f32) -> (u64, usize, usize) {
        let rate = 48000u32;
        let block = 3840usize; // 80 ms
        let start = block + (rate * SAFETY_MARGIN_MS / 1000) as usize;
        let stats = Arc::new(Stats::default());
        let (mut prod, cons) = RingBuffer::<f32>::new((rate * RING_SECONDS) as usize);
        let mut out = OutputState::new(cons, start, block, 2, rate, true, stats.clone());
        let mut data = vec![0.0f32; 480 * 2];
        let mut input_acc = 0.0f64;
        let mut max_level = 0;
        let mut first_underruns = None;
        let mut late_floor = usize::MAX;
        for tick in 0..60_000 {
            input_acc += 480.0 * (1.0 + drift);
            while input_acc >= block as f64 {
                for _ in 0..block {
                    let _ = prod.push(value);
                }
                input_acc -= block as f64;
            }
            out.fill(&mut data);
            if !out.priming && first_underruns.is_none() {
                first_underruns = Some(stats.underruns.load(Ordering::Relaxed));
            }
            max_level = max_level.max(out.ring.slots());
            if tick >= 59_500 {
                late_floor = late_floor.min(out.ring.slots());
            }
        }
        (stats.underruns.load(Ordering::Relaxed) - first_underruns.unwrap_or(0), max_level, late_floor)
    }

    /// In silence the output drops what the margin does not need, down to the cushion, and
    /// never underruns doing so; with sound it never drops anything.
    #[test]
    fn silence_trims_latency_to_the_cushion() {
        let rate = 48000usize;
        let margin = rate * SAFETY_MARGIN_MS as usize / 1000;
        let cushion = rate * MIN_CUSHION_MS as usize / 1000;
        let (underruns, _, floor) = simulate_with(0.0, 0.0);
        assert_eq!(underruns, 0);
        assert!(floor <= cushion + 480, "floor {floor} > cushion {cushion}");
        let (underruns, _, floor) = simulate_with(0.0, 0.5);
        assert_eq!(underruns, 0);
        assert!(floor >= margin - 480, "floor {floor} trimmed below the margin {margin} with sound");
    }

    #[test]
    fn output_ring_absorbs_bursts_and_drift() {
        let rate = 48000usize;
        let start = 3840 + rate * SAFETY_MARGIN_MS as usize / 1000;
        let bound = start + 3840 + rate / 20 + 3840;
        for drift in [0.0, 0.0001, 0.002, -0.0001, -0.002] {
            let (underruns, max_level) = simulate(drift);
            assert!(max_level <= bound, "drift {drift}: buffer grew to {max_level} (bound {bound})");
            if drift >= 0.0 {
                assert_eq!(underruns, 0, "drift {drift}: underruns with a fast or equal input clock");
            } else {
                // A slow input clock loses |drift| * 600 s of audio; each re-prime buys at least the
                // safety margin (the block itself is consumed while the next one is computed).
                let margin = rate as f64 * SAFETY_MARGIN_MS as f64 / 1000.0;
                let expected = (-drift * 600.0 * rate as f64 / margin).ceil() as u64 + 2;
                assert!(underruns <= expected, "drift {drift}: {underruns} underruns (expected <= {expected})");
            }
        }
    }
}
