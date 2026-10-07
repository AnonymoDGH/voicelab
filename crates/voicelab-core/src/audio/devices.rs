//! Audio device enumeration and selection (WASAPI on Windows, ALSA on Linux, CoreAudio on macOS).

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};

/// Output devices that act as a virtual microphone: whatever is played into them shows up as
/// a recording device in Discord/OBS/games. VB-Cable is the reference driver on Windows.
const VIRTUAL_CABLE_HINTS: [&str; 4] = ["cable input", "vb-audio", "voicemeeter input", "voicelab"];

pub const VB_CABLE_URL: &str = "https://vb-audio.com/Cable/";

#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceInfo {
    /// Stable identifier (pass it back to select the device).
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub is_virtual_cable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
}

fn describe(device: &cpal::Device) -> (String, String) {
    let id = device.id().map(|i| i.to_string()).unwrap_or_default();
    let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| id.clone());
    (id, name)
}

fn is_virtual_cable(name: &str) -> bool {
    let lower = name.to_lowercase();
    VIRTUAL_CABLE_HINTS.iter().any(|h| lower.contains(h))
}

pub fn list(direction: Direction) -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default_id = match direction {
        Direction::Input => host.default_input_device(),
        Direction::Output => host.default_output_device(),
    }
    .map(|d| describe(&d).0);
    let devices = match direction {
        Direction::Input => host.input_devices(),
        Direction::Output => host.output_devices(),
    };
    let Ok(devices) = devices else { return Vec::new() };
    devices
        .map(|d| {
            let (id, name) = describe(&d);
            DeviceInfo {
                is_default: default_id.as_deref() == Some(id.as_str()),
                is_virtual_cable: direction == Direction::Output && is_virtual_cable(&name),
                id,
                name,
            }
        })
        .collect()
}

/// The preferred virtual-cable output: VB-Cable's stereo "CABLE Input" first, then its 16-channel
/// endpoint or other virtual devices.
pub fn find_virtual_cable() -> Option<DeviceInfo> {
    let rank = |d: &DeviceInfo| {
        let n = d.name.to_lowercase();
        if n.starts_with("cable input") {
            0
        } else if n.contains("16ch") {
            2
        } else {
            1
        }
    };
    list(Direction::Output).into_iter().filter(|d| d.is_virtual_cable).min_by_key(rank)
}

/// Find a device by exact id, else by case-insensitive name substring; `None` = system default.
pub(crate) fn open(direction: Direction, wanted: Option<&str>) -> Result<(cpal::Device, String)> {
    let host = cpal::default_host();
    let device = match wanted {
        None => match direction {
            Direction::Input => host.default_input_device(),
            Direction::Output => host.default_output_device(),
        }
        .context("no hay dispositivo de audio por defecto")?,
        Some(w) => {
            let devices: Vec<cpal::Device> = match direction {
                Direction::Input => host.input_devices()?.collect(),
                Direction::Output => host.output_devices()?.collect(),
            };
            let lw = w.to_lowercase();
            let by_id = devices.iter().position(|d| describe(d).0 == w);
            let by_name = || devices.iter().position(|d| describe(d).1.to_lowercase().contains(&lw));
            let idx = by_id.or_else(by_name).with_context(|| format!("no encuentro el dispositivo «{w}»"))?;
            devices.into_iter().nth(idx).expect("index from position")
        }
    };
    let name = describe(&device).1;
    Ok((device, name))
}
