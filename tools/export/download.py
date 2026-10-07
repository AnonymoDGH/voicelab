"""Download the MeanVC2, WavLM speaker and AP-BWE checkpoints into ./ckpts."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from voicelab_export.paths import (
    BWE_FILE,
    BWE_GDRIVE_ID,
    BWE_HF,
    BWE_SHA256,
    HF_REPO,
    SPEAKER_FILE,
    VARIANTS,
    VOCODER_FILE,
    WAVLM_FINETUNE_GDRIVE_ID,
    WAVLM_FINETUNE_HF,
    WAVLM_FINETUNE_SHA256,
    ckpt_dir,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_verified(dst: Path, hf: tuple[str, str], gdrive_id: str, digest: str) -> None:
    if not dst.exists():
        try:
            print(f"[hf] {dst.name} (mirror)")
            shutil.copy(hf_hub_download(*hf), dst)
        except Exception as e:  # fall back to the original Google Drive upload
            import gdown

            print(f"[gdrive] {dst.name} ({e.__class__.__name__} on mirror)")
            gdown.download(id=gdrive_id, output=str(dst))
    if sha256(dst) != digest:
        dst.unlink()
        raise SystemExit(f"{dst.name}: SHA-256 mismatch, deleted; run again")


def main() -> None:
    out = ckpt_dir()
    out.mkdir(parents=True, exist_ok=True)
    names = {VOCODER_FILE}
    for v in VARIANTS.values():
        names |= {v.vc_file, v.asr_file}
    for name in sorted(names):
        dst = out / name
        if dst.exists():
            print(f"[skip] {dst}")
            continue
        print(f"[hf] {name}")
        shutil.copy(hf_hub_download(HF_REPO, name), dst)

    fetch_verified(out / SPEAKER_FILE, WAVLM_FINETUNE_HF, WAVLM_FINETUNE_GDRIVE_ID, WAVLM_FINETUNE_SHA256)
    fetch_verified(out / BWE_FILE, BWE_HF, BWE_GDRIVE_ID, BWE_SHA256)
    print("ok")


if __name__ == "__main__":
    main()
