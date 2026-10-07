//! Inverse STFT matching `torch.istft(center=True, window=hann_window(n_fft, periodic=True))`,
//! the last stage of the Vocos head (kept out of the ONNX graph).

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, num_complex::Complex32};

pub struct Istft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    frame: Vec<f32>,
}

impl Istft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let ifft = RealFftPlanner::<f32>::new().plan_fft_inverse(n_fft);
        let window = (0..n_fft)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos()) as f32)
            .collect();
        Self {
            n_fft,
            hop,
            window,
            spectrum: ifft.make_input_vec(),
            scratch: ifft.make_scratch_vec(),
            frame: ifft.make_output_vec(),
            ifft,
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// `real`/`imag` are `[bins, frames]` row-major (the Vocos ONNX output layout).
    /// Returns `hop * (frames - 1)` samples.
    pub fn process(&mut self, real: &[f32], imag: &[f32], frames: usize) -> Vec<f32> {
        let (n_fft, hop, bins) = (self.n_fft, self.hop, self.bins());
        assert_eq!(real.len(), bins * frames);
        let total = n_fft + hop * (frames - 1);
        let mut y = vec![0.0f32; total];
        let mut env = vec![0.0f32; total];
        let scale = 1.0 / n_fft as f32;
        for t in 0..frames {
            for b in 0..bins {
                self.spectrum[b] = Complex32::new(real[b * frames + t], imag[b * frames + t]);
            }
            // irfft ignores the imaginary part of the DC and Nyquist bins.
            self.spectrum[0].im = 0.0;
            self.spectrum[bins - 1].im = 0.0;
            self.ifft
                .process_with_scratch(&mut self.spectrum, &mut self.frame, &mut self.scratch)
                .expect("fft sizes are fixed");
            let off = t * hop;
            for i in 0..n_fft {
                let w = self.window[i];
                y[off + i] += self.frame[i] * scale * w;
                env[off + i] += w * w;
            }
        }
        let pad = n_fft / 2;
        (pad..total - pad).map(|i| y[i] / env[i]).collect()
    }
}
