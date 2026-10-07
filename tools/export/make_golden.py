"""Golden fixtures for the Rust engine, produced by the numpy/ONNX reference pipeline.

    uv run python make_golden.py --models ../../models --source in.wav --target ref.wav --out ../../tests/fixtures

Per variant: input audio, the exact ODE start noise of every DiT step, and the intermediates
(fbank per chunk, BN per ASR call, mel per DiT step, vocoder output, final audio). The Rust
tests replay the same noise and compare stage by stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from voicelab_export.onnx_pipeline import CHUNK, OnnxVC, fbank, istft_center, mel_banks, speaker_embedding
from voicelab_export.speaker import load_wav_16k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../../models")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out", default="../../tests/fixtures")
    args = ap.parse_args()
    models, out = Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((models / "manifest.json").read_text())

    wav = load_wav_16k(args.source)[: int(16000 * args.seconds)]
    target = load_wav_16k(args.target)[: 16000 * 4]
    emb = speaker_embedding(models, target, manifest["speaker"]["file"])

    # Stand-alone DSP fixtures (no models needed to check them)
    rng = np.random.default_rng(7)
    real = rng.standard_normal((321, 14)).astype(np.float32) * 3
    imag = rng.standard_normal((321, 14)).astype(np.float32) * 3
    save_file({
        "fbank_in": wav[:CHUNK + 160].copy(),
        "fbank_out": fbank(wav[:CHUNK + 160]),
        "mel_banks": mel_banks(),
        "istft_real": real,
        "istft_imag": imag,
        "istft_out": istft_center(real, imag),
    }, str(out / "dsp.safetensors"))

    save_file({"wav": target, "emb": emb.astype(np.float32)}, str(out / "speaker.safetensors"))

    for variant in manifest["variants"]:
        vc = OnnxVC(models, variant, threads=1, rng=np.random.default_rng(11))
        vc.set_speaker(emb)
        vc.trace = []
        out_wav = vc.process_stream(wav)
        t = {"input": wav, "spk": emb.astype(np.float32), "output": out_wav}
        by_kind: dict[str, list] = {}
        for e in vc.trace:
            by_kind.setdefault(e["kind"], []).append(e)
        t["fbank"] = np.concatenate([e["fbank"] for e in by_kind["fbank"]])
        t["fbank_frames"] = np.array([len(e["fbank"]) for e in by_kind["fbank"]], dtype=np.int64)
        t["bn"] = np.concatenate([e["bn"] for e in by_kind["asr"]])
        t["noise"] = np.stack([e["x0"] for e in by_kind["dit"]])
        t["cond"] = np.stack([e["cond"] for e in by_kind["dit"]])
        t["mel"] = np.stack([e["mel"] for e in by_kind["dit"]])
        t["k_mem"], t["v_mem"] = vc.k_mem[0], vc.v_mem[0]
        t = {k: np.ascontiguousarray(v) for k, v in t.items()}
        save_file(t, str(out / f"golden_{variant}.safetensors"))
        print(f"[golden] {variant}: {len(by_kind['dit'])} DiT steps, {len(by_kind['asr'])} ASR calls, "
              f"{len(out_wav)} output samples")


if __name__ == "__main__":
    main()
