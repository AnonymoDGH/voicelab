"""Export every MeanVC2 component to ONNX and check each graph against PyTorch.

    uv run python export_onnx.py --out ../../models

Writes `<out>/manifest.json` with file hashes and the streaming constants the engine needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from voicelab_export.graphs import GTM, DiTStep, SpeakerEmbed, VocosSpec
from voicelab_export.paths import SPEAKER_FILE, VARIANTS, VOCODER_FILE, ckpt
from voicelab_export.reference import load_vc_model
from voicelab_export.speaker import load_speaker_encoder

OPSET = 17


def export(module, args, path: Path, inputs, outputs, dynamic_axes=None) -> None:
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                          dynamic_axes=dynamic_axes, opset_version=OPSET, do_constant_folding=True)
    print(f"[onnx] {path.name} {path.stat().st_size / 1e6:.1f} MB")


def check(path: Path, module, feeds: dict[str, torch.Tensor], tol: float) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})
    with torch.no_grad():
        want = module(*feeds.values())
    want = want if isinstance(want, (tuple, list)) else (want,)
    # error relative to the output's scale (Vocos spectra reach 100, BN features ~1)
    err = max(float(np.abs(g - w.numpy()).max() / max(1.0, float(w.abs().max()))) for g, w in zip(got, want))
    status = "ok" if err <= tol else "FAIL"
    print(f"[check] {path.name}: max rel err {err:.2e} ({status})")
    if err > tol:
        raise SystemExit(f"{path.name} diverges from PyTorch")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../models")
    ap.add_argument("--skip-speaker", action="store_true", help="skip the 1.3 GB speaker encoder")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    manifest: dict = {"format": 1, "sample_rate": 16000, "variants": {}, "files": {}}

    # --- Vocoder -----------------------------------------------------------
    vocos = torch.jit.load(ckpt(VOCODER_FILE)).eval()
    voc = torch.jit.script(VocosSpec(vocos).eval())  # Vocos ships as TorchScript; trace through a scripted wrapper
    mel = torch.rand(1, 80, 14)
    export(voc, (mel,), out / "vocos.onnx", ["mel"], ["real", "imag"], {"mel": {2: "frames"}, "real": {2: "frames"},
                                                                        "imag": {2: "frames"}})
    check(out / "vocos.onnx", voc, {"mel": torch.rand(1, 80, 6)}, 1e-4)
    istft = vocos.head.istft
    manifest["vocoder"] = {"file": "vocos.onnx", "n_fft": int(istft.n_fft), "hop": int(istft.hop_length),
                           "win": int(istft.win_length), "window": "hann_periodic", "center": True}

    # --- Per-variant ASR, DiT, GTM ------------------------------------------
    for name, v in VARIANTS.items():
        asr = torch.jit.load(ckpt(v.asr_file)).eval()
        asr_feeds = {
            "fbank": torch.randn(1, v.bn_window, 80) * 3,
            "offset": torch.tensor(v.asr_offset_init + 40, dtype=torch.int64),
            "required_cache_size": torch.tensor(v.asr_cache, dtype=torch.int64),
            "att_cache": torch.randn(6, 4, v.asr_cache, 128),
            "cnn_cache": torch.randn(6, 1, 256, 8),
        }
        asr_path = out / f"asr_{name}.onnx"
        export(asr, tuple(asr_feeds.values()), asr_path, list(asr_feeds), ["bn", "att_cache_out", "cnn_cache_out"])
        check(asr_path, asr, asr_feeds, 1e-3)

        dit = load_vc_model(v)
        step = DiTStep(dit, v.chunk_size, v.block_size).eval()
        spk = torch.randn(1, 256) * 0.2
        with torch.no_grad():
            k_mem, v_mem = dit.gtm(spk)
        w = step.window
        dit_feeds = {
            "x": torch.randn(1, w, 80),
            "cond": torch.randn(1, w, 256),
            "spk": spk,
            "k_mem": k_mem,
            "v_mem": v_mem,
            "kv": torch.randn(*step.kv_shape()),
            "kv_valid": torch.tensor(v.chunk_size, dtype=torch.int64),
        }
        dit_path = out / f"dit_{name}.onnx"
        export(step, tuple(dit_feeds.values()), dit_path, list(dit_feeds), ["mel", "kv_out"])
        check(dit_path, step, dit_feeds, 1e-3)
        check(dit_path, step, {**dit_feeds, "kv_valid": torch.tensor(0, dtype=torch.int64)}, 1e-3)

        gtm = GTM(dit).eval()
        gtm_path = out / f"gtm_{name}.onnx"
        export(gtm, (spk,), gtm_path, ["spk"], ["k_mem", "v_mem"])
        check(gtm_path, gtm, {"spk": torch.randn(1, 256) * 0.2}, 1e-4)

        manifest["variants"][name] = {
            "asr": asr_path.name, "dit": dit_path.name, "gtm": gtm_path.name,
            "chunk_size": v.chunk_size, "block_size": v.block_size,
            "bn_window": v.bn_window, "bn_stride": v.bn_stride, "asr_cache": v.asr_cache,
            "asr_offset_init": v.asr_offset_init, "asr_offset_step": v.asr_offset_step,
            "kv_shape": list(step.kv_shape()), "cache_frames": step.cache_frames,
        }

    # --- Speaker encoder ----------------------------------------------------
    if not args.skip_speaker:
        enc = SpeakerEmbed(load_speaker_encoder(ckpt(SPEAKER_FILE))).eval()
        spk_path = out / "spk_encoder.onnx"
        export(enc, (torch.randn(1, 16000 * 3) * 0.1,), spk_path, ["wav"], ["emb"], {"wav": {1: "samples"}})
        check(spk_path, enc, {"wav": torch.randn(1, 16000 * 5) * 0.1}, 2e-3)
        manifest["speaker"] = {"file": spk_path.name, "sample_rate": 16000, "emb_dim": 256}

    for p in sorted(out.glob("*.onnx")):
        manifest["files"][p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
