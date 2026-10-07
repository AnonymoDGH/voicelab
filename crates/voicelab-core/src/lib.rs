//! VoiceLab core: real-time voice conversion on CPU (MeanVC2 on ONNX Runtime).

pub mod audio_file;
pub mod dsp;
pub mod engine;
pub mod paths;
pub mod voice;

pub use engine::{ModelDir, SpeakerEncoder, StreamingVc, Variant};
pub use voice::Voice;
