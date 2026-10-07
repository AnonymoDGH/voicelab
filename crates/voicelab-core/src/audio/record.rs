//! Record a reference clip from a microphone (for cloning a voice in-app).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};

use super::devices::{self, Direction};

/// Blocks for `seconds` and returns (mono samples, sample rate).
pub fn record(input: Option<&str>, seconds: f32) -> Result<(Vec<f32>, u32)> {
    let (dev, _) = devices::open(Direction::Input, input)?;
    let cfg = dev.default_input_config()?;
    let rate = cfg.sample_rate();
    let ch = cfg.channels() as usize;
    let buf = Arc::new(Mutex::new(Vec::with_capacity((rate as f32 * seconds) as usize)));
    let err = |e: cpal::Error| eprintln!("error de grabación: {e}");

    fn cb<T: SizedSample>(buf: Arc<Mutex<Vec<f32>>>, ch: usize) -> impl FnMut(&[T], &cpal::InputCallbackInfo) + Send
    where
        f32: FromSample<T>,
    {
        move |data: &[T], _| {
            if let Ok(mut b) = buf.lock() {
                b.extend(
                    data.chunks_exact(ch).map(|f| f.iter().map(|s| f32::from_sample(*s)).sum::<f32>() / ch as f32),
                );
            }
        }
    }
    let config = cfg.config();
    let stream = match cfg.sample_format() {
        SampleFormat::F32 => dev.build_input_stream(config, cb::<f32>(buf.clone(), ch), err, None)?,
        SampleFormat::I16 => dev.build_input_stream(config, cb::<i16>(buf.clone(), ch), err, None)?,
        SampleFormat::I32 => dev.build_input_stream(config, cb::<i32>(buf.clone(), ch), err, None)?,
        SampleFormat::U16 => dev.build_input_stream(config, cb::<u16>(buf.clone(), ch), err, None)?,
        other => bail!("formato de micrófono no soportado: {other}"),
    };
    stream.play()?;
    std::thread::sleep(Duration::from_secs_f32(seconds));
    drop(stream);
    let samples = std::mem::take(&mut *buf.lock().unwrap());
    Ok((samples, rate))
}
