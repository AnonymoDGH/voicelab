"""Download the MeanVC2 checkpoints (HF) and the WavLM speaker checkpoint (Google Drive) into ./ckpts."""

from __future__ import annotations

import shutil

from huggingface_hub import hf_hub_download

from voicelab_export.paths import HF_REPO, SPEAKER_FILE, VARIANTS, VOCODER_FILE, WAVLM_FINETUNE_GDRIVE_ID, ckpt_dir


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
        import gdown

        print(f"[gdrive] {SPEAKER_FILE}")
        gdown.download(id=WAVLM_FINETUNE_GDRIVE_ID, output=str(spk))
    print("ok")


if __name__ == "__main__":
    main()
