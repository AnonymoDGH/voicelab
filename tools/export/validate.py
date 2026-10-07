"""Phase 0: run the PyTorch reference on CPU (1 thread), report RTF and write converted wavs.

    uv run python validate.py --source es.wav --target ref.wav --variant 40ms --out out/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voicelab_export.paths import SPEAKER_FILE, VARIANTS, ckpt
from voicelab_export.reference import ReferenceVC
from voicelab_export.speaker import load_speaker_encoder, load_wav_16k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", required=True)
    ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--variant", nargs="+", default=["40ms", "120ms"])
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enc = load_speaker_encoder(ckpt(SPEAKER_FILE))
    embs = {}
    for t in args.target:
        t0 = time.perf_counter()
        with torch.no_grad():
            embs[t] = enc(torch.from_numpy(load_wav_16k(t)).unsqueeze(0))
        print(f"[spk] {Path(t).name}: {time.perf_counter() - t0:.2f}s norm={embs[t].norm():.2f}")

    for vname in args.variant:
        for t, emb in embs.items():
            ref = ReferenceVC(VARIANTS[vname], emb)
            for s in args.source:
                wav = load_wav_16k(s)
                ref.reset()
                torch.manual_seed(0)
                t0 = time.perf_counter()
                y = ref.process_stream(wav)
                dt = time.perf_counter() - t0
                name = f"{Path(s).stem}__to__{Path(t).stem}__{vname}.wav"
                sf.write(out / name, np.clip(y, -1, 1), 16000)
                print(f"[{vname}] {name}: {len(wav) / 16000:.1f}s audio, RTF {dt / (len(wav) / 16000):.3f}")


if __name__ == "__main__":
    main()
