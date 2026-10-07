"""Dynamic int8 quantization (weights int8, activations quantized on the fly).

    uv run python quantize.py --models ../../models [--only dit_120ms.onnx ...]

By default the speaker encoder and the Ultra bandwidth extension are quantized:
- speaker encoder 1.3 GB -> 386 MB, embedding cosine >= 0.998 vs fp32;
- AP-BWE 119 MB -> 41 MB and 2.7x faster; the voice band stays 34 dB SNR from fp32 (closer than
  fp32 is to its own input), the generated band above 8 kHz differs in fine detail only.
The other streaming graphs stay fp32 — they already run at RTF ~0.1 on one core and int8 Vocos
measurably lowered speaker similarity (bench_onnx.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from export_onnx import sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../../models")
    ap.add_argument("--only", nargs="*", default=["spk_encoder.onnx", "bwe_16k_48k.onnx"])
    args = ap.parse_args()
    models = Path(args.models)
    manifest = json.loads((models / "manifest.json").read_text())
    for name in args.only:
        src = models / name
        pre = models / f"{src.stem}.pre.onnx"
        dst = models / f"{src.stem}.int8.onnx"
        try:
            quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        except Exception as e:  # pre-processing is an optimization; quantize the raw graph if it fails
            print(f"[pre] {name}: skipped ({e.__class__.__name__})")
            pre = src
        quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                         extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
        if pre != src:
            pre.unlink()
        manifest["files"][dst.name] = {"sha256": sha256(dst), "bytes": dst.stat().st_size}
        print(f"[int8] {dst.name} {dst.stat().st_size / 1e6:.1f} MB (from {src.stat().st_size / 1e6:.1f} MB)")
        for part in ("speaker", "bwe"):
            if name == manifest.get(part, {}).get("file"):
                manifest[part]["file"] = dst.name
    (models / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
