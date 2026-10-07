//! Tauri backend: exposes the VoiceLab engine to the web UI.

mod settings;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use base64::prelude::{BASE64_STANDARD, Engine};
use serde::Serialize;
use settings::Settings;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
use voicelab_core::audio::devices::{self, DeviceInfo, Direction};
use voicelab_core::audio::{Controls, LiveConfig, LiveEngine, LiveInfo, StatsSnapshot, VB_CABLE_URL, record};
use voicelab_core::engine::download;
use voicelab_core::engine::vc::SAMPLE_RATE;
use voicelab_core::voice::{self, Voice};
use voicelab_core::{ModelDir, SpeakerEncoder, audio_file, paths};

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Default)]
struct Inner {
    settings: Settings,
    engine: Option<LiveEngine>,
    models: Option<ModelDir>,
}

pub struct AppState {
    inner: Mutex<Inner>,
    controls: Arc<Controls>,
}

impl AppState {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

fn apply_controls(c: &Controls, s: &Settings) {
    c.gate_db.store(s.gate_db);
    c.input_gain.store(s.input_gain);
    c.output_gain.store(s.output_gain);
    c.monitor.store(s.monitor_enabled, Ordering::Relaxed);
}

fn open_models() -> Option<ModelDir> {
    let dir = paths::models_dir();
    download::is_complete(&dir).then(|| ModelDir::open(&dir).ok()).flatten()
}

#[derive(Serialize)]
struct VoiceDto {
    id: String,
    name: String,
    description: String,
    source: String,
    license: String,
    builtin: bool,
    print: Vec<f32>,
    /// The voice's picture as a `data:` URL, ready for an `<img>` (the CSP allows `data:`).
    portrait: Option<String>,
    /// The picture is the user's own, so it can be removed.
    custom_portrait: bool,
}

impl VoiceDto {
    fn new(v: Voice, builtin: bool, user_dir: &Path) -> Self {
        let portrait = v.portrait(user_dir);
        Self {
            builtin,
            print: v.fingerprint(),
            custom_portrait: portrait.as_ref().is_some_and(|p| p.custom),
            portrait: portrait.and_then(|p| data_url(&p.path)),
            id: v.id,
            name: v.name,
            description: v.description,
            source: v.source,
            license: v.license,
        }
    }
}

/// A portrait file as a `data:` URL; `None` if unreadable or too big to be a thumbnail.
fn data_url(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?;
    let (_, mime) = voice::PORTRAIT_TYPES.iter().find(|(e, _)| e.eq_ignore_ascii_case(ext))?;
    let mut bytes = Vec::new();
    std::fs::File::open(path).ok()?.take(voice::PORTRAIT_MAX_BYTES + 1).read_to_end(&mut bytes).ok()?;
    (bytes.len() as u64 <= voice::PORTRAIT_MAX_BYTES)
        .then(|| format!("data:{mime};base64,{}", BASE64_STANDARD.encode(bytes)))
}

fn voices() -> Vec<VoiceDto> {
    let builtin = paths::builtin_voices_dir();
    let user_dir = paths::user_voices_dir();
    voice::list_voices(&paths::voice_dirs())
        .into_iter()
        .map(|v| {
            let is_builtin = match (&builtin, &v.path) {
                (Some(b), Some(p)) => p.starts_with(b),
                _ => false,
            };
            VoiceDto::new(v, is_builtin, &user_dir)
        })
        .collect()
}

fn find_voice(id: &str) -> CmdResult<Voice> {
    voice::list_voices(&paths::voice_dirs())
        .into_iter()
        .find(|v| v.id == id)
        .ok_or_else(|| format!("no existe la voz «{id}»"))
}

#[derive(Serialize)]
struct Overview {
    models_ready: bool,
    models_dir: String,
    model_source: String,
    voices: Vec<VoiceDto>,
    inputs: Vec<DeviceInfo>,
    outputs: Vec<DeviceInfo>,
    virtual_cable: Option<DeviceInfo>,
    vb_cable_url: &'static str,
    settings: Settings,
    running: bool,
    info: Option<LiveInfo>,
}

#[tauri::command]
fn overview(state: State<'_, AppState>) -> Overview {
    let inner = state.lock();
    Overview {
        models_ready: inner.models.is_some(),
        models_dir: paths::models_dir().display().to_string(),
        model_source: download::source(),
        voices: voices(),
        inputs: devices::list(Direction::Input),
        outputs: devices::list(Direction::Output),
        virtual_cable: devices::find_virtual_cable(),
        vb_cable_url: VB_CABLE_URL,
        settings: inner.settings.clone(),
        running: inner.engine.is_some(),
        info: inner.engine.as_ref().map(|e| e.info().clone()),
    }
}

#[derive(Serialize)]
struct LiveStatus {
    running: bool,
    enabled: bool,
    muted: bool,
    stats: Option<StatsSnapshot>,
    error: Option<String>,
}

#[tauri::command]
fn live_status(state: State<'_, AppState>) -> LiveStatus {
    let mut inner = state.lock();
    let mut error = inner.engine.as_ref().and_then(|e| e.last_error());
    if inner.engine.as_ref().is_some_and(|e| !e.is_running()) {
        error = error.or(Some("el motor de audio se detuvo".into()));
        inner.engine = None;
    }
    LiveStatus {
        running: inner.engine.is_some(),
        enabled: state.controls.enabled.load(Ordering::Relaxed),
        muted: state.controls.muted.load(Ordering::Relaxed),
        stats: inner.engine.as_ref().map(|e| e.stats()),
        error,
    }
}

fn start_blocking(app: &AppHandle) -> CmdResult<LiveInfo> {
    let state = app.state::<AppState>();
    let (models, settings) = {
        let mut inner = state.lock();
        inner.engine = None; // stop a previous engine first: it holds the devices
        let models = inner.models.clone().ok_or("faltan los modelos: descárgalos en Ajustes")?;
        (models, inner.settings.clone())
    };
    let voice_id = settings.voice.clone().ok_or("elige una voz primero")?;
    let voice = find_voice(&voice_id)?;
    apply_controls(&state.controls, &settings);
    // Open a monitor stream only when it is a different device than the main output.
    let monitor = settings.monitor.clone().filter(|m| Some(m) != settings.output.as_ref());
    let cfg = LiveConfig {
        input: settings.input.clone(),
        output: settings.output.clone(),
        monitor: monitor.or_else(|| {
            devices::list(Direction::Output).into_iter().find(|d| d.is_default && !d.is_virtual_cable).map(|d| d.id)
        }),
        variant: Some(settings.variant),
        threads: settings.threads,
    };
    let engine =
        LiveEngine::start(&models, cfg, &voice.embedding, state.controls.clone()).map_err(|e| format!("{e:#}"))?;
    let info = engine.info().clone();
    state.lock().engine = Some(engine);
    Ok(info)
}

#[tauri::command]
async fn start_engine(app: AppHandle) -> CmdResult<LiveInfo> {
    tauri::async_runtime::spawn_blocking(move || start_blocking(&app)).await.map_err(err)?
}

#[tauri::command]
fn stop_engine(state: State<'_, AppState>) {
    let engine = state.lock().engine.take();
    drop(engine);
}

#[tauri::command]
fn set_voice(state: State<'_, AppState>, id: String) -> CmdResult<()> {
    let voice = find_voice(&id)?;
    let mut inner = state.lock();
    if let Some(engine) = &inner.engine {
        engine.set_voice(voice.embedding);
    }
    inner.settings.voice = Some(id);
    inner.settings.save();
    Ok(())
}

#[tauri::command]
fn set_flags(state: State<'_, AppState>, enabled: Option<bool>, muted: Option<bool>, monitor: Option<bool>) {
    let c = &state.controls;
    if let Some(v) = enabled {
        c.enabled.store(v, Ordering::Relaxed);
    }
    if let Some(v) = muted {
        c.muted.store(v, Ordering::Relaxed);
    }
    if let Some(v) = monitor {
        c.monitor.store(v, Ordering::Relaxed);
        let mut inner = state.lock();
        inner.settings.monitor_enabled = v;
        inner.settings.save();
    }
}

/// Save settings. Live values (gains, gate) apply at once; device or model changes restart
/// the engine if it is running.
#[tauri::command]
async fn update_settings(app: AppHandle, settings: Settings) -> CmdResult<()> {
    let state = app.state::<AppState>();
    let restart = {
        let mut inner = state.lock();
        let old = &inner.settings;
        let needs_restart = old.input != settings.input
            || old.output != settings.output
            || old.monitor != settings.monitor
            || old.variant != settings.variant
            || old.threads != settings.threads;
        let hotkey_changed = old.hotkey != settings.hotkey;
        apply_controls(&state.controls, &settings);
        settings.save();
        inner.settings = settings.clone();
        if hotkey_changed {
            register_hotkey(&app, &settings.hotkey);
        }
        needs_restart && inner.engine.is_some()
    };
    if restart {
        tauri::async_runtime::spawn_blocking(move || start_blocking(&app)).await.map_err(err)??;
    }
    Ok(())
}

fn save_cloned(models: &ModelDir, wav16k: &[f32], name: &str, description: &str, source: &str) -> CmdResult<VoiceDto> {
    let name = name.trim();
    if name.is_empty() {
        return Err("ponle un nombre a la voz".into());
    }
    let emb = SpeakerEncoder::new(models, 4).and_then(|mut e| e.embed(wav16k)).map_err(|e| format!("{e:#}"))?;
    let mut v = Voice::new(voice::slug(name), name, emb);
    v.description = description.trim().to_string();
    v.source = source.to_string();
    v.license = "propia".into();
    // Ids are unique across all voices, built-in ones included: pictures are matched by id.
    let taken: Vec<String> = voice::list_voices(&paths::voice_dirs()).into_iter().map(|v| v.id).collect();
    let dir = paths::user_voices_dir();
    let mut path = dir.join(format!("{}.{}", v.id, voice::EXTENSION));
    let mut n = 2;
    while path.exists() || taken.contains(&v.id) {
        v.id = format!("{}-{n}", voice::slug(name));
        path = dir.join(format!("{}.{}", v.id, voice::EXTENSION));
        n += 1;
    }
    v.save(&path).map_err(err)?;
    v.path = Some(path);
    Ok(VoiceDto::new(v, false, &dir))
}

fn models_of(app: &AppHandle) -> CmdResult<ModelDir> {
    app.state::<AppState>().lock().models.clone().ok_or_else(|| "faltan los modelos: descárgalos en Ajustes".into())
}

#[tauri::command]
async fn clone_voice_file(app: AppHandle, path: PathBuf, name: String, description: String) -> CmdResult<VoiceDto> {
    let models = models_of(&app)?;
    tauri::async_runtime::spawn_blocking(move || {
        let wav = audio_file::read_mono_at(&path, SAMPLE_RATE).map_err(|e| format!("{e:#}"))?;
        let source = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        save_cloned(&models, &wav, &name, &description, &source)
    })
    .await
    .map_err(err)?
}

#[tauri::command]
async fn record_and_clone(app: AppHandle, seconds: f32, name: String, description: String) -> CmdResult<VoiceDto> {
    let models = models_of(&app)?;
    let input = app.state::<AppState>().lock().settings.input.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (wav, rate) = record::record(input.as_deref(), seconds.clamp(3.0, 30.0)).map_err(|e| format!("{e:#}"))?;
        let wav = voicelab_core::dsp::resample::resample_clip(&wav, rate, SAMPLE_RATE).map_err(err)?;
        save_cloned(&models, &wav, &name, &description, "grabación")
    })
    .await
    .map_err(err)?
}

#[tauri::command]
fn delete_voice(state: State<'_, AppState>, id: String) -> CmdResult<()> {
    let v = find_voice(&id)?;
    let path = v.path.ok_or("voz sin archivo")?;
    let user_dir = paths::user_voices_dir();
    if !path.starts_with(&user_dir) {
        return Err("las voces incluidas no se pueden borrar".into());
    }
    // Picture first: if anything fails, the voice is still there.
    voice::clear_portrait(&user_dir, &v.id).map_err(err)?;
    std::fs::remove_file(path).map_err(err)?;
    let mut inner = state.lock();
    if inner.settings.voice.as_deref() == Some(id.as_str()) {
        inner.settings.voice = None;
        inner.settings.save();
    }
    Ok(())
}

/// Gives a voice the user's own picture, sent as a `data:image/…;base64,…` URL (the UI crops
/// and shrinks it first). It replaces the built-in illustration until removed.
#[tauri::command]
fn set_voice_portrait(id: String, data_url: String) -> CmdResult<()> {
    let v = find_voice(&id)?;
    let (mime, data) =
        data_url.strip_prefix("data:").and_then(|s| s.split_once(";base64,")).ok_or("la imagen no es válida")?;
    let (ext, _) = voice::PORTRAIT_TYPES.iter().find(|(_, m)| *m == mime).ok_or("formato de imagen no admitido")?;
    let bytes = BASE64_STANDARD.decode(data).map_err(|_| "la imagen no es válida")?;
    voice::set_portrait(&paths::user_voices_dir(), &v.id, &bytes, ext).map_err(err)?;
    Ok(())
}

/// Removes the user's own picture of a voice (built-in voices get their illustration back).
#[tauri::command]
fn clear_voice_portrait(id: String) -> CmdResult<()> {
    let v = find_voice(&id)?;
    voice::clear_portrait(&paths::user_voices_dir(), &v.id).map_err(err)
}

#[derive(Clone, Serialize)]
struct DownloadEvent {
    done: u64,
    total: u64,
    file: String,
}

#[tauri::command]
async fn download_models(app: AppHandle) -> CmdResult<()> {
    let dir = paths::models_dir();
    let handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut last = 0u64;
        download::download(&dir, &download::source(), |p, file| {
            // ~1 event per MB keeps the UI smooth without flooding it.
            if p.done - last >= 1 << 20 || p.done == p.total {
                last = p.done;
                let _ =
                    handle.emit("download-progress", DownloadEvent { done: p.done, total: p.total, file: file.into() });
            }
        })
        .map_err(|e| format!("{e:#}"))?;
        let models = ModelDir::open(&dir).map_err(err)?;
        app.state::<AppState>().lock().models = Some(models);
        Ok(())
    })
    .await
    .map_err(err)?
}

fn register_hotkey(app: &AppHandle, hotkey: &str) {
    let gs = app.global_shortcut();
    let _ = gs.unregister_all();
    let _ = gs.on_shortcut(hotkey, |app, _shortcut, event| {
        if event.state() == ShortcutState::Pressed {
            let c = &app.state::<AppState>().controls;
            let enabled = !c.enabled.load(Ordering::Relaxed);
            c.enabled.store(enabled, Ordering::Relaxed);
            let _ = app.emit("enabled-changed", enabled);
        }
    });
}

pub fn run() {
    let settings = Settings::load();
    let controls = Arc::new(Controls::default());
    apply_controls(&controls, &settings);
    let hotkey = settings.hotkey.clone();
    let state = AppState { inner: Mutex::new(Inner { settings, engine: None, models: open_models() }), controls };

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .manage(state)
        .setup(move |app| {
            register_hotkey(app.handle(), &hotkey);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            overview,
            live_status,
            start_engine,
            stop_engine,
            set_voice,
            set_flags,
            update_settings,
            clone_voice_file,
            record_and_clone,
            delete_voice,
            set_voice_portrait,
            clear_voice_portrait,
            download_models,
        ])
        .run(tauri::generate_context!())
        .expect("error al iniciar VoiceLab");
}
