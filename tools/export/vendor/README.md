# Vendored third-party code

Used only by the export tooling (never shipped in the app).

| Directory | Source | License |
|-----------|--------|---------|
| `meanvc2/` | [ASLP-lab/MeanVC2](https://github.com/ASLP-lab/MeanVC2) `runtime/src/{dit,dit_modules,modules}.py`, `src/config/*.json` | Apache-2.0 |
| `wavlm/` | [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm) `wavlm/{WavLM,modules}.py` | MIT, Copyright (c) Microsoft Corporation |

Local changes: `wavlm/WavLM.py` imports `modules` relatively. Nothing else is modified.
