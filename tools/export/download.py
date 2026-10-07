"""Download the MeanVC2 checkpoints and the WavLM speaker checkpoint into ./ckpts."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from voicelab_export.paths import (
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

    spk = out / SPEAKER_FILE
    if not spk.exists():
        try:
            print(f"[hf] {SPEAKER_FILE} (mirror)")
            shutil.copy(hf_hub_download(*WAVLM_FINETUNE_HF), spk)
        except Exception as e:  # fall back to the original Google Drive upload
            import gdown

            print(f"[gdrive] {SPEAKER_FILE} ({e.__class__.__name__} on mirror)")
            gdown.download(id=WAVLM_FINETUNE_GDRIVE_ID, output=str(spk))
    if sha256(spk) != WAVLM_FINETUNE_SHA256:
        spk.unlink()
        raise SystemExit(f"{SPEAKER_FILE}: SHA-256 mismatch, deleted; run again")
    print("ok")


if __name__ == "__main__":
    main()
