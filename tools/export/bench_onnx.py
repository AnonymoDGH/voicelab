"""Benchmark the ONNX pipeline (RTF, per-chunk latency) and write converted wavs.

    uv run python bench_onnx.py --models ../../models --source in.wav --target ref.wav --int8 dit vocos
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voicelab_export.onnx_pipeline import CHUNK, SR, OnnxVC, speaker_embedding
from voicelab_export.speaker import load_wav_16k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../../models")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--variant", nargs="+", default=["120ms", "40ms"])
    ap.add_argument("--int8", nargs="*", default=[], help="components to run in int8: asr dit vocos spk")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    models, out = Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spk_file = "spk_encoder.int8.onnx" if "spk" in args.int8 else "spk_encoder.onnx"
    t0 = time.perf_counter()
    emb = speaker_embedding(models, load_wav_16k(args.target), spk_file)
    print(f"[spk] {spk_file}: {time.perf_counter() - t0:.2f}s")

    wav = load_wav_16k(args.source)
    tag = "int8-" + "-".join(args.int8) if args.int8 else "fp32"
    for variant in args.variant:
        vc = OnnxVC(models, variant, threads=args.threads, int8=tuple(args.int8))
        vc.set_speaker(emb)
        times, parts = [], []
        for pos in range(0, len(wav), CHUNK):
            t0 = time.perf_counter()
            o = vc.process_chunk(wav[pos:pos + CHUNK])
            times.append(time.perf_counter() - t0)
            if o is not None:
                parts.append(o)
        ms = np.array(times[2:]) * 1000  # skip warm-up chunks
        rtf = sum(times) / (len(wav) / SR)
        name = f"{Path(args.source).stem}__to__{Path(args.target).stem}__{variant}__{tag}.wav"
        sf.write(out / name, np.clip(np.concatenate(parts), -1, 1), SR)
        print(f"[{variant}/{tag}] RTF {rtf:.3f}  chunk(160ms) p50 {np.median(ms):.1f} ms  p99 {np.percentile(ms, 99):.1f} ms  -> {name}")


if __name__ == "__main__":
    main()
