//! Reading (wav/mp3/flac/ogg) and writing (wav) audio files as mono f32.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::dsp::resample::resample_clip;

/// Decode any supported file to mono f32 at its native rate. Returns (samples, sample_rate).
pub fn read_mono(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| format!("unsupported audio format: {}", path.display()))?;
    let mut format = probed.format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).context("no audio track")?;
    let track_id = track.id;
    let rate = track.codec_params.sample_rate.context("unknown sample rate")?;
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut mono = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let sb = buf.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if sb.capacity() < decoded.capacity() * channels {
            *sb = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        sb.copy_interleaved_ref(decoded);
        mono.extend(sb.samples().chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
    }
    if mono.is_empty() {
        bail!("{} contains no audio", path.display());
    }
    Ok((mono, rate))
}

/// Decode and resample to `rate`.
pub fn read_mono_at(path: impl AsRef<Path>, rate: u32) -> Result<Vec<f32>> {
    let (samples, sr) = read_mono(path)?;
    resample_clip(&samples, sr, rate)
}

pub fn write_wav(path: impl AsRef<Path>, samples: &[f32], rate: u32) -> Result<()> {
    let spec =
        hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path.as_ref(), spec)
        .with_context(|| format!("creating {}", path.as_ref().display()))?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}
