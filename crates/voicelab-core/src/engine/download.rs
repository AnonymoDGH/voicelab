//! First-run model download, with size and SHA-256 checks. By default the models come from the
//! GitHub release matching this version (the release workflow exports and attaches them).
//! Only the files the engine uses are fetched (the fp32 speaker encoder is skipped).

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use super::models::{Manifest, Variant};

const RELEASES: &str = "https://github.com/AnonymoDGH/voicelab/releases/download";

/// Base URL the model files are fetched from (`VOICELAB_MODEL_URL` overrides it, e.g. a
/// `https://huggingface.co/<repo>/resolve/main/` mirror).
pub fn source() -> String {
    std::env::var("VOICELAB_MODEL_URL").unwrap_or_else(|_| format!("{RELEASES}/v{}/", env!("CARGO_PKG_VERSION")))
}

/// HTTP client that trusts the OS certificate store, so downloads work behind corporate proxies
/// and antivirus HTTPS inspection (their root CA is installed in the system, not in webpki).
fn agent() -> ureq::Agent {
    use ureq::tls::{RootCerts, TlsConfig};
    ureq::Agent::config_builder()
        .tls_config(TlsConfig::builder().root_certs(RootCerts::PlatformVerifier).build())
        .build()
        .into()
}

fn url(base: &str, file: &str) -> String {
    format!("{}/{file}", base.trim_end_matches('/'))
}

/// Files the engine needs, from a manifest.
pub fn required_files(m: &Manifest) -> BTreeSet<String> {
    let mut files: BTreeSet<String> =
        m.variants.values().flat_map(|v| [v.asr.clone(), v.dit.clone(), v.gtm.clone()]).collect();
    files.insert(m.vocoder.file.clone());
    files.insert(m.speaker.file.clone());
    files.extend(m.bwe.iter().map(|b| b.file.clone()));
    files
}

/// True when `dir` has a manifest with every mode and every required file with the right size.
/// Model sets from before Ultra mode count as incomplete, so the app fetches the missing graphs
/// (files already present with the right size are not downloaded again).
pub fn is_complete(dir: &Path) -> bool {
    let Ok(text) = fs::read_to_string(dir.join("manifest.json")) else { return false };
    let Ok(m) = serde_json::from_str::<Manifest>(&text) else { return false };
    let all_modes = [Variant::Quality, Variant::LowLatency, Variant::Ultra]
        .iter()
        .all(|v| m.variants.contains_key(v.key()) && (!v.uses_bwe() || m.bwe.is_some()));
    all_modes
        && required_files(&m).iter().all(|f| {
            let expected = m.files.get(f).map(|e| e.bytes);
            fs::metadata(dir.join(f)).is_ok_and(|md| Some(md.len()) == expected)
        })
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
}

/// Download (or resume after a failure) every required file into `dir`.
pub fn download(dir: &Path, base: &str, mut progress: impl FnMut(Progress, &str)) -> Result<()> {
    fs::create_dir_all(dir)?;
    let http = agent();
    let manifest_text = http
        .get(&url(base, "manifest.json"))
        .call()
        .with_context(|| format!("no se pudo descargar el manifiesto de {base}"))?
        .body_mut()
        .read_to_string()?;
    let manifest: Manifest = serde_json::from_str(&manifest_text).context("manifiesto inválido")?;
    let files = required_files(&manifest);
    let total: u64 = files.iter().filter_map(|f| manifest.files.get(f)).map(|e| e.bytes).sum();
    let mut done = 0u64;

    for file in &files {
        let entry = manifest.files.get(file).with_context(|| format!("{file} no está en el manifiesto"))?;
        let dst = dir.join(file);
        if fs::metadata(&dst).is_ok_and(|m| m.len() == entry.bytes) {
            done += entry.bytes;
            progress(Progress { done, total }, file);
            continue;
        }
        let part = dir.join(format!("{file}.part"));
        let mut resp = http.get(&url(base, file)).call().with_context(|| format!("descargando {file}"))?;
        let mut reader = resp.body_mut().with_config().limit(entry.bytes + 1).reader();
        let mut out = File::create(&part)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            got += n as u64;
            progress(Progress { done: done + got, total }, file);
        }
        out.flush()?;
        drop(out);
        let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if got != entry.bytes || digest != entry.sha256 {
            let _ = fs::remove_file(&part);
            bail!("{file}: descarga corrupta (tamaño o SHA-256 no coinciden)");
        }
        fs::rename(&part, &dst)?;
        done += entry.bytes;
    }
    // The manifest goes last: its presence marks a complete install.
    fs::write(dir.join("manifest.json"), manifest_text)?;
    Ok(())
}
