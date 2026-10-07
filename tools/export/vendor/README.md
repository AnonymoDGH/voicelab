# Vendored third-party code

Used only by the export tooling (never shipped in the app).

| Directory | Source | License |
|-----------|--------|---------|
| `meanvc2/` | [ASLP-lab/MeanVC2](https://github.com/ASLP-lab/MeanVC2) `runtime/src/{dit,dit_modules,modules}.py`, `src/config/*.json` | Apache-2.0 |
| `apbwe/` | [yxlu-0102/AP-BWE](https://github.com/yxlu-0102/AP-BWE) `models/model.py` (generator only) | MIT, Copyright (c) 2023 Ye-Xin Lu |
| `wavlm/` | [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm) `wavlm/{WavLM,modules}.py` | MIT, Copyright (c) Microsoft Corporation |

Local changes: `wavlm/WavLM.py` imports `modules` relatively; `apbwe/model.py` keeps only the
generator and inlines `get_padding`. Nothing else is modified.
