"""Checkpoint locations and per-variant streaming constants (from MeanVC2 runtime/run_rt.py)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HF_REPO = "ASLP-lab/MeanVC2"
WAVLM_FINETUNE_GDRIVE_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
# Byte-identical mirror of the UniSpeech checkpoint (Google Drive rate-limits CI downloads).
WAVLM_FINETUNE_HF = ("bezzam/wavlm_large_finetune_seed_tts_eval", "wavlm_large_finetune.pth")
WAVLM_FINETUNE_SHA256 = "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b"
VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "meanvc2"

# AP-BWE 16 kHz -> 48 kHz bandwidth extension (Ultra mode). Byte-identical HF mirror of the
# authors' Google Drive release; the Drive id is the fallback.
BWE_FILE = "g_16kto48k"
BWE_HF = ("rsxdalv/AP-BWE", "weights/16kto48k/g_16kto48k.zip")
BWE_GDRIVE_ID = "1HYkD_5ha9GMrjzbTiSFiO1uQQwxXET15"
BWE_SHA256 = "305a05dcab7dc29ffba09d32692d7a34550fc8fbdf338013641ff5d39a3cb285"
# configs/config_16kto48k.json
BWE_CONFIG = {"n_fft": 1024, "hop_size": 80, "win_size": 320, "ConvNeXt_channels": 512, "ConvNeXt_layers": 8}
# Mean-flow steps of the Ultra variant (2 for the others). 4 steps raise no-reference PESQ by ~0.3
# over 2 at the same speaker similarity; 8 adds little more for twice the cost.
ULTRA_STEPS = 4


def ckpt_dir() -> Path:
    return Path(os.environ.get("VOICELAB_CKPTS", Path(__file__).resolve().parents[1] / "ckpts"))


@dataclass(frozen=True)
class Variant:
    name: str
    vc_file: str
    asr_file: str
    config: Path
    chunk_size: int
    block_size: int
    bn_window: int
    bn_stride: int
    asr_cache: int
    asr_offset_init: int
    asr_offset_step: int


VARIANTS = {
    "120ms": Variant("120ms", "meanvc2_120ms_40ms.safetensors", "fastu2pp_160ms.pt", VENDOR / "config_120ms_40ms.json",
                     12, 4, 19, 16, 8, 8, 4),
    "40ms": Variant("40ms", "meanvc2_40ms_40ms.safetensors", "fastu2pp_80ms.pt", VENDOR / "config_40ms_40ms.json",
                    4, 4, 11, 8, 4, 4, 2),
}

VOCODER_FILE = "vocos.pt"
SPEAKER_FILE = "wavlm_large_finetune.pth"


def ckpt(name: str) -> str:
    return str(ckpt_dir() / name)
