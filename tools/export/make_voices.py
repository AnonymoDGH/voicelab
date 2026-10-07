"""Built-in voices: speaker embeddings of permissively licensed clips, saved as `.vlvoice`.

    uv run python make_voices.py --models ../../models --out ../../voices

A `.vlvoice` file is a safetensors file with one tensor, `spk_emb` (f32[256]), and string
metadata (name, description, source, license). It is independent of the model variant: the
engine derives the timbre memory (GTM) from the embedding when the voice is selected.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file

from voicelab_export.onnx_pipeline import speaker_embedding
from voicelab_export.speaker import load_wav_16k

VCTK = ("VCTK Corpus (CSTR, University of Edinburgh) via kyutai/tts-voices", "CC-BY-4.0")
DONATION = ("Unmute Voice Donation Project (Kyutai) via kyutai/tts-voices", "CC0-1.0")
UNMUTE = ("Kyutai's own recordings for unmute.sh via kyutai/tts-voices", "CC0-1.0")
VOICES = [
    # id, file in kyutai/tts-voices, display name, description[, (source, license)]
    ("lucia", "vctk/p228_023_enhanced.wav", "Lucía", "Femenina · VCTK p228"),
    ("carlos", "vctk/p254_023_enhanced.wav", "Carlos", "Masculina · VCTK p254"),
    ("elena", "vctk/p244_023_enhanced.wav", "Elena", "Femenina · VCTK p244"),
    ("pablo", "vctk/p259_023_enhanced.wav", "Pablo", "Masculina · VCTK p259"),
    ("marta", "vctk/p333_023_enhanced.wav", "Marta", "Femenina · VCTK p333"),
    ("diego", "vctk/p360_023_enhanced.wav", "Diego", "Masculina · VCTK p360"),
    ("sofia", "vctk/p229_023_enhanced.wav", "Sofía", "Femenina · VCTK p229"),
    ("jorge", "vctk/p315_023_enhanced.wav", "Jorge", "Masculina · VCTK p315"),
    # Volunteers who donated their voice for cloning (CC0); picked for clean audio, Spanish
    # speakers first. Display names are personas, not the donors' names.
    ("valeria", "voice-donations/Andrea_enhanced.wav", "Valeria", "Femenina · español · donada", DONATION),
    ("mateo", "voice-donations/Yesid_enhanced.wav", "Mateo", "Masculina · español latino · donada", DONATION),
    ("rafael", "voice-donations/Richard_cuban_enhanced.wav", "Rafael", "Masculina · grave, acento cubano · donada", DONATION),
    ("ivan", "voice-donations/spanish-limaperu_enhanced.wav", "Iván", "Masculina · español de Perú · donada", DONATION),
    ("alvaro", "voice-donations/Enrique_(Spanish)_enhanced.wav", "Álvaro", "Masculina · español de España · donada", DONATION),
    ("clara", "voice-donations/Hannah_enhanced.wav", "Clara", "Femenina · inglés · donada", DONATION),
    ("irene", "voice-donations/Lara_enhanced.wav", "Irene", "Femenina · inglés · donada", DONATION),
    ("noelia", "unmute-prod-website/developpeuse-3.wav", "Noelia", "Femenina · francés", UNMUTE),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../../models")
    ap.add_argument("--out", default="../../voices")
    ap.add_argument("--only", nargs="*", help="voice ids to (re)generate; default all")
    args = ap.parse_args()
    models, out = Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spk_file = json.loads((models / "manifest.json").read_text())["speaker"]["file"]
    with tempfile.TemporaryDirectory() as tmp:
        for vid, path, name, desc, *origin in VOICES:
            if args.only and vid not in args.only:
                continue
            source, license_ = origin[0] if origin else VCTK
            wav = load_wav_16k(hf_hub_download("kyutai/tts-voices", path, cache_dir=tmp))
            emb = speaker_embedding(models, wav, spk_file).astype(np.float32)
            meta = {"name": name, "description": desc, "source": f"{source}: {path}", "license": license_}
            save_file({"spk_emb": emb}, str(out / f"{vid}.vlvoice"), metadata=meta)
            print(f"[voice] {vid}.vlvoice  {name} — {desc}")


if __name__ == "__main__":
    main()
