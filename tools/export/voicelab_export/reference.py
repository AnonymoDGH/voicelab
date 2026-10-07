"""PyTorch reference of the MeanVC2 streaming pipeline (port of runtime/run_rt.py VCRunner).

This is the ground truth the ONNX graphs and the Rust engine are checked against.
`trace` collects per-step intermediates for golden fixtures; `noise_fn` lets a caller
inject the ODE start noise so another implementation can be compared bit-for-bit.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

from vendor.meanvc2.dit import DiT
from voicelab_export.paths import VOCODER_FILE, Variant, ckpt

CHUNK = 2560  # 160 ms at 16 kHz per process_chunk call
ODE_STEPS = 2
VOCODER_OVERLAP = 2
UPSAMPLE = 160


def load_vc_model(variant: Variant) -> DiT:
    from safetensors.torch import load_file

    cfg = json.loads(variant.config.read_text())
    model = DiT(**cfg["model"])
    model.load_state_dict(load_file(ckpt(variant.vc_file)), strict=False)
    return model.float().eval()


def kaldi_fbank(samples: np.ndarray) -> torch.Tensor:
    wav = torch.from_numpy(samples * (1 << 15)).unsqueeze(0)
    return kaldi.fbank(wav, frame_length=25, frame_shift=10, snip_edges=True, num_mel_bins=80,
                       energy_floor=0.0, dither=0.0, sample_frequency=16000)


class ReferenceVC:
    def __init__(self, variant: Variant, spk_emb: torch.Tensor):
        torch.set_num_threads(1)
        self.v = variant
        self.asr = torch.jit.load(ckpt(variant.asr_file)).eval()
        self.vc = load_vc_model(variant)
        self.vocoder = torch.jit.load(ckpt(VOCODER_FILE)).eval()
        self.chunk_size = variant.chunk_size
        self.block_size = variant.block_size
        self.wav_overlap = UPSAMPLE
        self.wav_pad = (VOCODER_OVERLAP - 1) * UPSAMPLE - self.wav_overlap
        self.down = torch.linspace(1, 0, steps=self.wav_overlap).numpy()
        self.up = torch.linspace(0, 1, steps=self.wav_overlap).numpy()
        self.noise_fn: Callable[[int], torch.Tensor] = lambda t: torch.randn(1, t, 80)
        self.trace: list[dict] | None = None
        self.set_speaker(spk_emb)
        self.reset()

    def set_speaker(self, spk_emb: torch.Tensor) -> None:
        self.spk = spk_emb.reshape(1, 256).float()
        with torch.no_grad():
            self.gtm_kv = self.vc.gtm(self.spk)

    def reset(self) -> None:
        self.samples_cache = np.zeros(0, dtype=np.float32)
        self.frame_cache = None
        self.encoder_output_cache = None
        self.asr_offset = self.v.asr_offset_init
        self.asr_att_cache = torch.zeros(6, 4, self.v.asr_cache, 128)
        self.asr_cnn_cache = torch.zeros(6, 1, 256, 8)
        self.vc_kv_cache = None
        self.vc_offset = 0
        self.noise_cache = None
        self.vocoder_cache = None
        self.last_wav = None
        self.bn_buffer = None
        self.min_bn_len = self.chunk_size + self.block_size

    def _log(self, **kw) -> None:
        if self.trace is not None:
            self.trace.append({k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()})

    @torch.no_grad()
    def encode_chunk(self, samples: np.ndarray) -> torch.Tensor | None:
        padded = np.concatenate((self.samples_cache, samples)) if self.samples_cache.size else samples
        if len(padded) < 400:
            self.samples_cache = padded
            return None
        fbanks = kaldi_fbank(padded)
        self._log(kind="fbank", samples=torch.from_numpy(padded.copy()), fbank=fbanks)
        self.samples_cache = padded[160 * fbanks.shape[0]:]
        if self.frame_cache is not None:
            fbanks = torch.cat([self.frame_cache, fbanks], dim=0)
        if self.asr_offset >= 4000:
            self.asr_offset = max(self.asr_att_cache.size(2), self.asr_offset - 4000)
        bns = []
        i = 0
        while i + self.v.bn_window <= fbanks.shape[0]:
            fb = fbanks[i:i + self.v.bn_window].unsqueeze(0)
            att_in, cnn_in = self.asr_att_cache, self.asr_cnn_cache
            out, self.asr_att_cache, self.asr_cnn_cache = self.asr(
                fb, torch.tensor(self.asr_offset, dtype=torch.int64),
                torch.tensor(self.v.asr_cache, dtype=torch.int64), att_in, cnn_in)
            self._log(kind="asr", fbank=fb, offset=self.asr_offset, att_cache=att_in, cnn_cache=cnn_in,
                      bn=out, att_cache_out=self.asr_att_cache, cnn_cache_out=self.asr_cnn_cache)
            bns.append(out.squeeze(0))
            self.asr_offset += self.v.asr_offset_step
            i += self.v.bn_stride
        self.frame_cache = fbanks[i:]
        if not bns:
            return None
        bn = torch.cat(bns, dim=0).unsqueeze(0)
        if self.encoder_output_cache is not None:
            bn = torch.cat([self.encoder_output_cache, bn], dim=1)
        self.encoder_output_cache = bn[:, -1:, :]
        if bn.shape[1] < 2:
            return None
        up = F.interpolate(bn.transpose(1, 2), size=17, mode="linear", align_corners=True).transpose(1, 2)
        return up[:, 1:, :]

    @torch.no_grad()
    def vc_step(self, cond: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / ODE_STEPS
        x = self.noise_fn(cond.shape[1]).clone()
        if self.noise_cache is not None:
            x[:, :self.block_size, :] = self.noise_cache
        self.noise_cache = x[:, -self.block_size:, :].clone()
        if self.vc_offset >= 4000:
            self.vc_offset = 0
            self.vc_kv_cache = None
        pre = self.vc_kv_cache
        x0, offset = x.clone(), self.vc_offset
        for i in range(ODE_STEPS):
            t = 1.0 - i * dt
            r = max(0.0, t - dt)
            u, kv = self.vc(x, torch.full((1,), t), torch.full((1,), r), cache=None, cond=cond, spks=self.spk,
                            offset=self.vc_offset, is_inference=True, kv_cache=pre, gtm_kv=self.gtm_kv)
            x = x - dt * u
            if i == ODE_STEPS - 1:
                self.vc_kv_cache = kv
        self.vc_offset += self.chunk_size
        mel = x[:, :self.chunk_size, :].transpose(1, 2)
        self._log(kind="dit", x0=x0, cond=cond, offset=offset, kv_in=pre, kv_out=self.vc_kv_cache, mel=mel)
        return mel

    @torch.no_grad()
    def decode_mel(self, mel: torch.Tensor) -> np.ndarray:
        if self.vocoder_cache is not None:
            mel = torch.cat([self.vocoder_cache, mel], dim=-1)
        self.vocoder_cache = mel[:, :, -VOCODER_OVERLAP:]
        voc_in = (mel + 1) / 2
        wav = self.vocoder.decode(voc_in).squeeze().numpy()
        self._log(kind="vocoder", mel=voc_in, wav=torch.from_numpy(wav.copy()))
        if self.last_wav is not None:
            wav = wav[self.wav_pad:]
            front = wav[:self.wav_overlap]
            new_wav = np.concatenate([self.last_wav * self.down + front * self.up,
                                      wav[self.wav_overlap:-self.wav_overlap]])
        else:
            new_wav = wav[:-self.wav_overlap]
        self.last_wav = wav[-self.wav_overlap:]
        return new_wav

    def process_chunk(self, samples: np.ndarray) -> np.ndarray | None:
        bn_new = self.encode_chunk(samples)
        if bn_new is None:
            return None
        self.bn_buffer = bn_new if self.bn_buffer is None else torch.cat([self.bn_buffer, bn_new], dim=1)
        parts = []
        for _ in range(4):
            if self.bn_buffer.shape[1] < self.min_bn_len:
                break
            cond = self.bn_buffer[:, :self.min_bn_len, :]
            self.bn_buffer = self.bn_buffer[:, self.chunk_size:, :]
            parts.append(self.decode_mel(self.vc_step(cond)))
        return np.concatenate(parts) if parts else None

    def process_stream(self, wav: np.ndarray) -> np.ndarray:
        """Feed `wav` in CHUNK blocks exactly like the realtime path (no tail flush)."""
        out = []
        for pos in range(0, len(wav), CHUNK):
            o = self.process_chunk(wav[pos:pos + CHUNK].astype(np.float32))
            if o is not None:
                out.append(o)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
