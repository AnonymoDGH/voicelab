//! Ultra mode: AP-BWE bandwidth extension, 16 kHz speech -> 48 kHz, streamed.
//!
//! The model takes the converted voice upsampled to 48 kHz (so nothing above 8 kHz) as STFT
//! log-magnitude and phase (n_fft 1024, hann window 320, hop 80, `torch.stft(center=True)`
//! framing) and predicts the full-band spectrum. It is a stack of kernel-7 convolutions over
//! time with per-frame norms and linears, so an output frame depends on `context` frames on each
//! side and nothing else. Every call runs the model on the frames whose receptive field just
//! became complete plus that context, and overlap-adds them: the result is the same as
//! processing the whole utterance at once (`tools/export` `OnnxBwe`), apart from the stream
//! start, which is zero-padded instead of reflected.

use std::sync::Arc;

use anyhow::Result;
use ort::session::Session;
use ort::value::TensorRef;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex, num_complex::Complex32};

use super::models::{BweSpec, ModelDir};
use super::vc::{SAMPLE_RATE, run_err};
use crate::dsp::stream_resample::StreamResampler;

pub const OUTPUT_RATE: u32 = 48000;

pub struct Bwe {
    session: Session,
    n_fft: usize,
    hop: usize,
    win: usize,
    context: usize,
    bins: usize,
    log_floor: f32,
    /// The `win`-long hann window; it sits centered in the n_fft frame, zeros elsewhere.
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    frame: Vec<f32>,
    spectrum: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    ifft_scratch: Vec<Complex32>,
    /// Input samples from absolute index `input_start` (negative = zero padding before the stream).
    input: Vec<f32>,
    input_start: i64,
    /// `[frame][bin]` log-magnitude and phase of frames `feat_start..next_frame`.
    log_amp: Vec<f32>,
    phase: Vec<f32>,
    feat_start: i64,
    next_frame: i64,
    /// Next frame to synthesize.
    next_out: i64,
    /// Overlap-add signal and squared-window envelope from absolute sample `emitted`.
    ola: Vec<f32>,
    env: Vec<f32>,
    emitted: i64,
    // Model I/O, `[bin][frame]`.
    in_amp: Vec<f32>,
    in_pha: Vec<f32>,
    out_amp: Vec<f32>,
    out_re: Vec<f32>,
    out_im: Vec<f32>,
}

impl Bwe {
    pub fn new(models: &ModelDir, threads: usize) -> Result<Self> {
        let spec: &BweSpec = models.manifest.bwe.as_ref().ok_or_else(|| anyhow::anyhow!("no bwe in manifest"))?;
        anyhow::ensure!(spec.sample_rate == OUTPUT_RATE, "bwe must output {OUTPUT_RATE} Hz");
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(spec.n_fft);
        let ifft = planner.plan_fft_inverse(spec.n_fft);
        let window = (0..spec.win)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / spec.win as f64).cos()) as f32)
            .collect();
        let mut bwe = Self {
            session: models.session(&spec.file, threads)?,
            n_fft: spec.n_fft,
            hop: spec.hop,
            win: spec.win,
            context: spec.context_frames,
            bins: spec.n_fft / 2 + 1,
            log_floor: spec.log_floor,
            window,
            frame: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            fft_scratch: fft.make_scratch_vec(),
            ifft_scratch: ifft.make_scratch_vec(),
            fft,
            ifft,
            input: Vec::new(),
            input_start: 0,
            log_amp: Vec::new(),
            phase: Vec::new(),
            feat_start: 0,
            next_frame: 0,
            next_out: 0,
            ola: Vec::new(),
            env: Vec::new(),
            emitted: 0,
            in_amp: Vec::new(),
            in_pha: Vec::new(),
            out_amp: Vec::new(),
            out_re: Vec::new(),
            out_im: Vec::new(),
        };
        bwe.reset();
        Ok(bwe)
    }

    pub fn reset(&mut self) {
        let half = (self.win / 2) as i64;
        self.input.clear();
        self.input.resize(half as usize, 0.0);
        self.input_start = -half;
        self.log_amp.clear();
        self.phase.clear();
        self.feat_start = 0;
        self.next_frame = 0;
        self.next_out = 0;
        self.ola.clear();
        self.env.clear();
        self.emitted = 0;
    }

    /// Output delay in samples at 48 kHz, for input fed in multiples of the hop: `context`
    /// frames of lookahead plus the window overlap of the frames still open.
    pub fn latency_samples(&self) -> usize {
        (self.context - 1) * self.hop + self.win
    }

    /// A whole 16 kHz clip -> 48 kHz full band, time-aligned with the input.
    pub fn extend_clip(&mut self, wav16: &[f32]) -> Result<Vec<f32>> {
        let mut up = StreamResampler::new(SAMPLE_RATE, OUTPUT_RATE)?;
        let mut x48 = Vec::new();
        up.push(wav16, &mut x48);
        up.push(&[0.0; SAMPLE_RATE as usize / 10], &mut x48); // flush the resampler
        self.reset();
        let mut out = Vec::new();
        self.process(&x48, &mut out)?;
        self.process(&vec![0.0; self.latency_samples() + self.hop], &mut out)?;
        let n = wav16.len() * (OUTPUT_RATE / SAMPLE_RATE) as usize;
        Ok(out.into_iter().skip(up.delay()).take(n).collect())
    }

    /// Feed 48 kHz band-limited samples; appends the finished full-band samples to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        self.input.extend_from_slice(input);
        self.analyze();
        let ready = self.next_frame - self.context as i64; // frames with full right context
        if ready <= self.next_out {
            return Ok(());
        }
        let first = (self.next_out - self.context as i64).max(0); // first model input frame
        let t = (self.next_frame - first) as usize;
        let (bins, off) = (self.bins, (first - self.feat_start) as usize);
        self.in_amp.resize(bins * t, 0.0);
        self.in_pha.resize(bins * t, 0.0);
        for f in 0..t {
            for b in 0..bins {
                self.in_amp[b * t + f] = self.log_amp[(off + f) * bins + b];
                self.in_pha[b * t + f] = self.phase[(off + f) * bins + b];
            }
        }
        let shape = [1usize, bins, t];
        let res = self
            .session
            .run(ort::inputs![
                "log_amp" => TensorRef::from_array_view((shape, &self.in_amp[..])).map_err(run_err)?,
                "pha" => TensorRef::from_array_view((shape, &self.in_pha[..])).map_err(run_err)?,
            ])
            .map_err(run_err)?;
        for (dst, name) in
            [(&mut self.out_amp, "log_amp_wb"), (&mut self.out_re, "pha_re"), (&mut self.out_im, "pha_im")]
        {
            dst.clear();
            dst.extend_from_slice(res[name].try_extract_tensor::<f32>().map_err(run_err)?.1);
        }
        drop(res);
        for j in self.next_out..ready {
            let f = (j - first) as usize;
            for b in 0..bins {
                let (r, i) = (self.out_re[b * t + f], self.out_im[b * t + f]);
                let norm = (r * r + i * i).sqrt().max(1e-12);
                let a = self.out_amp[b * t + f].exp() / norm;
                self.spectrum[b] = Complex32::new(a * r, a * i);
            }
            self.spectrum[0].im = 0.0;
            self.spectrum[bins - 1].im = 0.0;
            self.ifft
                .process_with_scratch(&mut self.spectrum, &mut self.frame, &mut self.ifft_scratch)
                .expect("fft sizes are fixed");
            self.overlap_add(j);
        }
        self.next_out = ready;
        // Samples before the first still-open frame's window are final.
        let done = ready * self.hop as i64 - (self.win / 2) as i64;
        let n = (done - self.emitted).max(0) as usize;
        out.extend((0..n).map(|i| if self.env[i] > 1e-11 { self.ola[i] / self.env[i] } else { 0.0 }));
        self.ola.drain(..n);
        self.env.drain(..n);
        self.emitted += n as i64;
        // Keep `context` frames of features before the next output frame.
        let keep_from = (self.next_out - self.context as i64).max(self.feat_start);
        let drop_frames = (keep_from - self.feat_start) as usize;
        self.log_amp.drain(..drop_frames * bins);
        self.phase.drain(..drop_frames * bins);
        self.feat_start = keep_from;
        Ok(())
    }

    /// STFT of every frame whose window is fully inside the input received so far.
    fn analyze(&mut self) {
        let (hop, half) = (self.hop as i64, (self.win / 2) as i64);
        let left = (self.n_fft - self.win) / 2;
        let end = self.input_start + self.input.len() as i64;
        while self.next_frame * hop + half <= end {
            let start = (self.next_frame * hop - half - self.input_start) as usize;
            self.frame.fill(0.0);
            for (k, w) in self.window.iter().enumerate() {
                self.frame[left + k] = self.input[start + k] * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.fft_scratch)
                .expect("fft sizes are fixed");
            for c in &self.spectrum {
                self.log_amp.push((c.norm() + self.log_floor).ln());
                self.phase.push(c.im.atan2(c.re));
            }
            self.next_frame += 1;
        }
        // Drop input no future frame reads.
        let first_needed = self.next_frame * hop - half;
        let n = (first_needed - self.input_start).clamp(0, self.input.len() as i64) as usize;
        self.input.drain(..n);
        self.input_start += n as i64;
    }

    /// Add frame `j` (time signal in `self.frame`) to the output, windowed, at its position.
    fn overlap_add(&mut self, j: i64) {
        let (half, left) = ((self.win / 2) as i64, (self.n_fft - self.win) / 2);
        let start = j * self.hop as i64 - half; // absolute sample of the window's first tap
        let end = (start + self.win as i64 - self.emitted) as usize;
        if self.ola.len() < end {
            self.ola.resize(end, 0.0);
            self.env.resize(end, 0.0);
        }
        let scale = 1.0 / self.n_fft as f32;
        for (k, w) in self.window.iter().enumerate() {
            let s = start + k as i64;
            if s < self.emitted {
                continue; // before the stream start
            }
            let i = (s - self.emitted) as usize;
            self.ola[i] += self.frame[left + k] * scale * w;
            self.env[i] += w * w;
        }
    }
}
