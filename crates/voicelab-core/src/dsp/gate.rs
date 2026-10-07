//! Noise gate driven by the input level. The model turns silence into breathy noise, so the
//! converted output is muted while nobody talks. Opens instantly, closes after a hold time that
//! covers the model latency (so word endings are not cut), with short fades to avoid clicks.

pub struct Gate {
    sample_rate: f32,
    hold_samples: usize,
    held: usize,
    gain: f32,
    step: f32,
}

impl Gate {
    pub fn new(sample_rate: u32, hold_ms: f32, fade_ms: f32) -> Self {
        Self {
            sample_rate: sample_rate as f32,
            hold_samples: (hold_ms / 1000.0 * sample_rate as f32) as usize,
            held: 0,
            gain: 0.0,
            step: 1.0 / (fade_ms / 1000.0 * sample_rate as f32).max(1.0),
        }
    }

    /// Update with an input block's level (dBFS) and apply to an output block.
    /// `threshold_db <= -100` disables the gate.
    pub fn process(&mut self, input_db: f32, threshold_db: f32, out: &mut [f32]) {
        let open = threshold_db <= -100.0 || input_db >= threshold_db;
        for s in out.iter_mut() {
            // The hold counts from the end of the last block with speech.
            let target = if open || self.held > 0 { 1.0 } else { 0.0 };
            if !open {
                self.held = self.held.saturating_sub(1);
            }
            if self.gain < target {
                self.gain = (self.gain + self.step).min(1.0);
            } else if self.gain > target {
                self.gain = (self.gain - self.step).max(0.0);
            }
            *s *= self.gain;
        }
        if open {
            self.held = self.hold_samples;
        }
    }

    pub fn is_open(&self) -> bool {
        self.gain > 0.0
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

pub fn rms_db(x: &[f32]) -> f32 {
    if x.is_empty() {
        return -120.0;
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    20.0 * rms.max(1e-6).log10()
}

pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_on_speech_and_closes_after_hold() {
        let mut g = Gate::new(16000, 100.0, 5.0);
        let mut buf = vec![1.0f32; 1600];
        g.process(-80.0, -45.0, &mut buf);
        assert!(buf.iter().all(|&v| v == 0.0), "closed gate must mute");
        let mut buf = vec![1.0f32; 1600];
        g.process(-20.0, -45.0, &mut buf);
        assert!((buf[1599] - 1.0).abs() < 1e-6, "open gate passes audio after the fade");
        let mut buf = vec![1.0f32; 3200];
        g.process(-80.0, -45.0, &mut buf);
        assert!((buf[800] - 1.0).abs() < 1e-6, "hold keeps it open");
        assert_eq!(buf[3199], 0.0, "closes after hold + fade");
    }

    #[test]
    fn disabled_gate_passes_everything() {
        let mut g = Gate::new(16000, 100.0, 5.0);
        let mut buf = vec![0.5f32; 400];
        g.process(-120.0, -100.0, &mut buf);
        assert!((buf[399] - 0.5).abs() < 1e-6);
    }
}
