"""Checkpoint locations and per-variant streaming constants (from MeanVC2 runtime/run_rt.py)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HF_REPO = "ASLP-lab/MeanVC2"
WAVLM_FINETUNE_GDRIVE_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "meanvc2"


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
