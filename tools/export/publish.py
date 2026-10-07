"""Publish the exported models to the Hugging Face repo the app downloads from.

    HF_TOKEN=hf_... uv run python publish.py --models ../../models --repo VoidWalkercero/voicelab-models

Uploads only what the engine needs (the files listed by the manifest's variants, vocoder and
speaker entries), the manifest itself (last) and a model card with the licenses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: apache-2.0
tags: [voice-conversion, onnx, real-time, cpu]
---

# VoiceLab models

ONNX exports used by [VoiceLab](https://github.com/AnonymoDGH/voicelab), a real-time CPU voice changer.

| File | Source | License |
|---|---|---|
| `dit_*.onnx`, `gtm_*.onnx` | [MeanVC2](https://huggingface.co/ASLP-lab/MeanVC2) DiT mean-flow + timbre memory | Apache-2.0 |
| `asr_*.onnx` | Fast-U2++ (WeNet) encoder shipped with MeanVC2 | Apache-2.0 |
| `vocos.onnx` | Vocos vocoder shipped with MeanVC2 (head without iSTFT) | MIT |
| `spk_encoder.int8.onnx` | WavLM-Large + ECAPA-TDNN speaker verification (Microsoft UniSpeech), int8 | MIT |

`manifest.json` lists every file with its SHA-256 and the streaming constants the engine needs.
Use responsibly: clone only your own voice or voices you have permission to use.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../../models")
    ap.add_argument("--repo", default="VoidWalkercero/voicelab-models")
    args = ap.parse_args()
    models = Path(args.models)
    manifest = json.loads((models / "manifest.json").read_text())

    files = {manifest["vocoder"]["file"], manifest["speaker"]["file"]}
    for v in manifest["variants"].values():
        files |= {v["asr"], v["dit"], v["gtm"]}

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=args.repo)
    for name in sorted(files):
        print(f"[upload] {name}")
        api.upload_file(path_or_fileobj=str(models / name), path_in_repo=name, repo_id=args.repo)
    # The app treats the manifest as the "install complete" marker, so it goes last.
    api.upload_file(path_or_fileobj=str(models / "manifest.json"), path_in_repo="manifest.json", repo_id=args.repo)
    print(f"[done] https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
