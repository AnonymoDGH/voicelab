"""Export every MeanVC2 component to ONNX and check each graph against PyTorch.

    uv run python export_onnx.py --out ../../models [--only ultra bwe]

Writes `<out>/manifest.json` with file hashes and the streaming constants the engine needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from voicelab_export.graphs import GTM, Bwe, DiTStep, SpeakerEmbed, VocosSpec
from voicelab_export.paths import BWE_CONFIG, BWE_FILE, SPEAKER_FILE, ULTRA_STEPS, VARIANTS, VOCODER_FILE, ckpt
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
    ap.add_argument("--only", nargs="*", choices=PARTS, help="export these parts into an existing manifest")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    parts = set(args.only or PARTS) - ({"speaker"} if args.skip_speaker else set())
    if args.only:
        manifest = json.loads((out / "manifest.json").read_text())
    else:
        manifest = {"format": 1, "sample_rate": 16000, "variants": {}, "files": {}}

    if "vocoder" in parts:
        export_vocoder(out, manifest)
    if "variants" in parts:
        for name, v in VARIANTS.items():
            export_variant(out, manifest, name, v)
    if "ultra" in parts:
        # Quality variant with more mean-flow steps; shares the ASR and GTM graphs.
        export_variant(out, manifest, "ultra", VARIANTS["120ms"], steps=ULTRA_STEPS, dit_only=True)
    if "bwe" in parts:
        export_bwe(out, manifest)
    if "speaker" in parts:
        export_speaker(out, manifest)

    for p in sorted(out.glob("*.onnx")):
        manifest["files"][p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {out / 'manifest.json'}")


PARTS = ["vocoder", "variants", "ultra", "bwe", "speaker"]


def export_vocoder(out: Path, manifest: dict) -> None:
    vocos = torch.jit.load(ckpt(VOCODER_FILE)).eval()
    voc = torch.jit.script(VocosSpec(vocos).eval())  # Vocos ships as TorchScript; trace through a scripted wrapper
    mel = torch.rand(1, 80, 14)
    export(voc, (mel,), out / "vocos.onnx", ["mel"], ["real", "imag"], {"mel": {2: "frames"}, "real": {2: "frames"},
                                                                        "imag": {2: "frames"}})
    check(out / "vocos.onnx", voc, {"mel": torch.rand(1, 80, 6)}, 1e-4)
    istft = vocos.head.istft
    manifest["vocoder"] = {"file": "vocos.onnx", "n_fft": int(istft.n_fft), "hop": int(istft.hop_length),
                           "win": int(istft.win_length), "window": "hann_periodic", "center": True}


def export_variant(out: Path, manifest: dict, name: str, v, steps: int = 2, dit_only: bool = False) -> None:
    if not dit_only:
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
    step = DiTStep(dit, v.chunk_size, v.block_size, steps).eval()
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

    if dit_only:
        base = manifest["variants"][v.name]
        manifest["variants"][name] = {**base, "dit": dit_path.name, "ode_steps": steps}
        return

    gtm = GTM(dit).eval()
    gtm_path = out / f"gtm_{name}.onnx"
    export(gtm, (spk,), gtm_path, ["spk"], ["k_mem", "v_mem"])
    check(gtm_path, gtm, {"spk": torch.randn(1, 256) * 0.2}, 1e-4)

    manifest["variants"][name] = {
        "asr": asr_path.name, "dit": dit_path.name, "gtm": gtm_path.name,
        "chunk_size": v.chunk_size, "block_size": v.block_size,
        "bn_window": v.bn_window, "bn_stride": v.bn_stride, "asr_cache": v.asr_cache,
        "asr_offset_init": v.asr_offset_init, "asr_offset_step": v.asr_offset_step,
        "kv_shape": list(step.kv_shape()), "cache_frames": step.cache_frames, "ode_steps": steps,
    }


def export_bwe(out: Path, manifest: dict) -> None:
    from vendor.apbwe.model import APNet_BWE_Model

    class Cfg(dict):
        __getattr__ = dict.__getitem__

    model = APNet_BWE_Model(Cfg(BWE_CONFIG))
    model.load_state_dict(torch.load(ckpt(BWE_FILE), map_location="cpu", weights_only=True)["generator"])
    bwe = Bwe(model.eval()).eval()
    bins = BWE_CONFIG["n_fft"] // 2 + 1
    feeds = {"log_amp": torch.randn(1, bins, 150) * 2 - 4, "pha": (torch.rand(1, bins, 150) * 2 - 1) * math.pi}
    path = out / "bwe_16k_48k.onnx"
    export(bwe, tuple(feeds.values()), path, list(feeds), ["log_amp_wb", "pha_re", "pha_im"],
           {k: {2: "frames"} for k in [*feeds, "log_amp_wb", "pha_re", "pha_im"]})
    check(path, bwe, {k: v[:, :, :60] for k, v in feeds.items()}, 1e-3)
    manifest["bwe"] = {"file": path.name, "sample_rate": 48000, "n_fft": BWE_CONFIG["n_fft"],
                       "hop": BWE_CONFIG["hop_size"], "win": BWE_CONFIG["win_size"], "window": "hann_periodic",
                       "center": True, "log_floor": 1e-4,
                       # conv_pre (k7) + 8 ConvNeXt blocks (k7): 3 + 8 * 3 frames on each side
                       "context_frames": 3 + 3 * BWE_CONFIG["ConvNeXt_layers"]}


def export_speaker(out: Path, manifest: dict) -> None:
    enc = SpeakerEmbed(load_speaker_encoder(ckpt(SPEAKER_FILE))).eval()
    spk_path = out / "spk_encoder.onnx"
    export(enc, (torch.randn(1, 16000 * 3) * 0.1,), spk_path, ["wav"], ["emb"], {"wav": {1: "samples"}})
    check(spk_path, enc, {"wav": torch.randn(1, 16000 * 5) * 0.1}, 2e-3)
    manifest["speaker"] = {"file": spk_path.name, "sample_rate": 16000, "emb_dim": 256}


if __name__ == "__main__":
    main()
