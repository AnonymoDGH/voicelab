//! `voicelab` — línea de comandos del motor de VoiceLab.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use voicelab_core::engine::vc::{BLOCK, SAMPLE_RATE};
use voicelab_core::voice::{self, Voice};
use voicelab_core::{ModelDir, SpeakerEncoder, StreamingVc, Variant, audio_file, paths};

#[derive(Parser)]
#[command(name = "voicelab", version, about = "Cambiador de voz con IA en tiempo real, solo CPU")]
struct Cli {
    /// Carpeta de modelos (por defecto: VOICELAB_MODELS, carpeta de datos o ./models)
    #[arg(long, global = true)]
    models: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Lista las voces disponibles
    Voices,
    /// Convierte un archivo de audio a otra voz (mismo camino que el tiempo real)
    Convert {
        input: PathBuf,
        output: PathBuf,
        /// Voz: id/nombre, archivo .vlvoice o audio de referencia (wav/mp3/flac/ogg)
        #[arg(short, long)]
        voice: String,
        /// 120ms (calidad) o 40ms (baja latencia)
        #[arg(long, default_value = "120ms")]
        variant: Variant,
        #[arg(long, default_value_t = 1)]
        threads: usize,
    },
    /// Mide si tu CPU llega a tiempo real
    Bench {
        /// Audio de entrada (por defecto: señal sintética)
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, default_value_t = 20.0)]
        seconds: f32,
        #[arg(long, default_value_t = 1)]
        threads: usize,
    },
    /// Crea una voz nueva a partir de 5-20 s de audio limpio de una persona
    CloneVoice {
        reference: PathBuf,
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value = "")]
        description: String,
        /// Carpeta de salida (por defecto: carpeta de voces del usuario)
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let models_path = cli.models.clone().unwrap_or_else(paths::models_dir);
    let models = || {
        ModelDir::open(&models_path)
            .with_context(|| format!("no hay modelos en {} (usa --models o VOICELAB_MODELS)", models_path.display()))
    };
    match cli.cmd {
        Cmd::Voices => {
            let voices = voice::list_voices(&paths::voice_dirs());
            if voices.is_empty() {
                println!("No hay voces. Crea una con `voicelab clone-voice`.");
            }
            for v in voices {
                println!("{:<12} {:<10} {}", v.id, v.name, v.description);
            }
        }
        Cmd::Convert { input, output, voice, variant, threads } => {
            let models = models()?;
            let v = resolve_voice(&models, &voice)?;
            let wav = audio_file::read_mono_at(&input, SAMPLE_RATE)?;
            let mut vc = StreamingVc::new(&models, variant, threads)?;
            vc.set_speaker(&v.embedding)?;
            let t0 = Instant::now();
            let out = vc.convert_clip(&wav)?;
            let secs = wav.len() as f32 / SAMPLE_RATE as f32;
            audio_file::write_wav(&output, &out, SAMPLE_RATE)?;
            println!(
                "{} -> {} con la voz «{}» ({:?}): {secs:.1} s de audio en {:.2} s (RTF {:.3})",
                input.display(),
                output.display(),
                v.name,
                variant,
                t0.elapsed().as_secs_f32(),
                t0.elapsed().as_secs_f32() / secs
            );
        }
        Cmd::Bench { input, seconds, threads } => bench(&models()?, input.as_deref(), seconds, threads)?,
        Cmd::CloneVoice { reference, name, description, out } => {
            let models = models()?;
            let mut v = clone_voice(&models, &reference, &name)?;
            v.description = description;
            let dir = out.unwrap_or_else(paths::user_voices_dir);
            let path = dir.join(format!("{}.{}", v.id, voice::EXTENSION));
            v.save(&path)?;
            println!("Voz «{}» guardada en {}", v.name, path.display());
        }
    }
    Ok(())
}

fn clone_voice(models: &ModelDir, reference: &Path, name: &str) -> Result<Voice> {
    let wav = audio_file::read_mono_at(reference, SAMPLE_RATE)?;
    let t0 = Instant::now();
    let emb = SpeakerEncoder::new(models, 4)?.embed(&wav)?;
    eprintln!("Huella de voz calculada en {:.1} s", t0.elapsed().as_secs_f32());
    let mut v = Voice::new(voice::slug(name), name, emb);
    v.source = reference.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(v)
}

fn resolve_voice(models: &ModelDir, spec: &str) -> Result<Voice> {
    let path = Path::new(spec);
    if path.is_file() {
        if path.extension().is_some_and(|e| e == voice::EXTENSION) {
            return Voice::load(path);
        }
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        return clone_voice(models, path, &name);
    }
    let wanted = spec.to_lowercase();
    voice::list_voices(&paths::voice_dirs())
        .into_iter()
        .find(|v| v.id == wanted || v.name.to_lowercase() == wanted)
        .with_context(|| format!("no encuentro la voz «{spec}» (mira `voicelab voices`)"))
}

fn bench(models: &ModelDir, input: Option<&Path>, seconds: f32, threads: usize) -> Result<()> {
    let wav = match input {
        Some(p) => audio_file::read_mono_at(p, SAMPLE_RATE)?,
        None => synthetic_speechlike((seconds * SAMPLE_RATE as f32) as usize),
    };
    if wav.len() < BLOCK * 4 {
        bail!("el audio de prueba es demasiado corto");
    }
    let ms = |samples: usize| samples as f32 * 1000.0 / SAMPLE_RATE as f32;
    println!("{threads} hilo(s), {:.1} s de audio", ms(wav.len()) / 1000.0);
    for variant in [Variant::Quality, Variant::LowLatency] {
        let mut vc = StreamingVc::new(models, variant, threads)?;
        vc.set_speaker(&[0.05f32; voicelab_core::engine::vc::SPK_DIM])?;
        let block = vc.block_samples();
        for chunk in wav.chunks(block).take(4) {
            vc.process(chunk)?; // calentamiento
        }
        vc.reset();
        let mut times: Vec<f32> = wav
            .chunks_exact(block)
            .map(|chunk| {
                let t0 = Instant::now();
                vc.process(chunk).map(|_| t0.elapsed().as_secs_f32() * 1000.0)
            })
            .collect::<Result<_>>()?;
        let total: f32 = times.iter().sum();
        times.sort_by(f32::total_cmp);
        let pct = |p: f32| times[((times.len() - 1) as f32 * p) as usize];
        let budget = ms(block);
        let rtf = total / (times.len() as f32 * budget);
        let verdict = if pct(0.99) < budget * 0.7 {
            "OK, tiempo real con margen"
        } else if pct(0.99) < budget {
            "justo"
        } else {
            "NO llega a tiempo real"
        };
        println!(
            "{:<6} bloque {:>3.0} ms | RTF {rtf:.3} | p50 {:.1} ms, p99 {:.1} ms | latencia del modelo ~{:.0} ms + proceso p99 | {verdict}",
            variant.key(),
            budget,
            pct(0.5),
            pct(0.99),
            ms(vc.algorithmic_latency_samples()),
        );
    }
    Ok(())
}

/// Deterministic voice-like test signal (harmonics with a wandering pitch + noise bursts).
fn synthetic_speechlike(n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut seed = 0x2545F491u32;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let f0 = 140.0 + 40.0 * (t * 1.7).sin();
            phase += 2.0 * std::f32::consts::PI * f0 / SAMPLE_RATE as f32;
            let voiced: f32 = (1..8).map(|h| (phase * h as f32).sin() / h as f32).sum();
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let env = 0.5 + 0.5 * (t * 3.0).sin();
            0.15 * env * voiced + 0.01 * noise
        })
        .collect()
}
