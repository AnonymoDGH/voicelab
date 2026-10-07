"""Export-friendly wrappers around the MeanVC2 modules.

Every graph has static shapes so ONNX Runtime can plan memory once:

* DiTStep runs both mean-flow ODE steps of one streaming window. The per-layer KV cache
  is a fixed-size tensor holding the last `cache_frames` positions (left-padded while
  warming up); `kv_valid` says how many of them are real. Padding is masked out and RoPE
  uses positions relative to the start of the cache, which leaves every query/key
  distance identical to the original growing-cache implementation.
* VocosSpec stops before the inverse STFT; the engine does the iSTFT itself.
"""

from __future__ import annotations

import math

import torch
from torch import nn

ODE_STEPS = 2


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class DiTStep(nn.Module):
    def __init__(self, dit, chunk_size: int, block_size: int):
        super().__init__()
        for m in dit.modules():
            if hasattr(m, "native_rms_norm"):
                m.native_rms_norm = False  # aten::rms_norm has no ONNX symbolic in opset 17
        self.dit = dit
        self.chunk = chunk_size
        self.block = block_size
        self.window = chunk_size + block_size
        blocks = dit.transformer_blocks
        self.depth = len(blocks)
        self.heads = blocks[0].attn.heads
        self.head_dim = blocks[0].attn.inner_dim // self.heads
        t_ps = [b.attn.processor.t_p for b in blocks]
        self.cache_frames = (max(t_ps) + 1) * chunk_size
        total = self.cache_frames + self.window

        masks = torch.stack([b.attn.processor.try_cached_mask(total, "cpu")[-self.window:] for b in blocks])
        self.register_buffer("chunk_mask", masks, persistent=False)  # [L, W, total] bool

        pos = torch.arange(total, dtype=torch.float32)
        freqs = pos[:, None] * dit.rotary_embed.inv_freq[None, :]
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)  # [total, Dh]
        self.register_buffer("cos_k", freqs.cos(), persistent=False)
        self.register_buffer("sin_k", freqs.sin(), persistent=False)

        dt = 1.0 / ODE_STEPS
        with torch.no_grad():
            t_embs = []
            for i in range(ODE_STEPS):
                t = 1.0 - i * dt
                r = max(0.0, t - dt)
                t_embs.append(dit.t_time_embed(torch.full((1,), t)) + dit.r_time_embed(torch.full((1,), r)))
        self.register_buffer("t_embs", torch.stack(t_embs), persistent=False)  # [S, 1, dim]

    def kv_shape(self) -> tuple[int, ...]:
        return (self.depth, 2, 1, self.heads, self.cache_frames, self.head_dim)

    def _attention(self, block, x, layer, kv_k, kv_v, bias):
        attn = block.attn
        q = attn.to_q(x).view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        k = attn.to_k(x).view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        v = attn.to_v(x).view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        if attn.q_norm is not None:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        k_all = torch.cat([kv_k, k], dim=2)
        v_all = torch.cat([kv_v, v], dim=2)
        new_k = k_all[:, :, self.chunk:self.chunk + self.cache_frames]
        new_v = v_all[:, :, self.chunk:self.chunk + self.cache_frames]

        cos_q, sin_q = self.cos_k[-self.window:], self.sin_k[-self.window:]
        q = q * cos_q + _rotate_half(q) * sin_q
        k_all = k_all * self.cos_k + _rotate_half(k_all) * self.sin_k

        w = (q @ k_all.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim)) + bias[layer]
        out = torch.softmax(w, dim=-1) @ v_all
        out = out.transpose(1, 2).reshape(1, -1, self.heads * self.head_dim)
        return attn.to_out[0](out), new_k, new_v

    def forward(self, x, cond, spk, k_mem, v_mem, kv, kv_valid):
        """
        x:        [1, W, 80]   start noise (first `block` frames carry the previous window's tail)
        cond:     [1, W, 256]  BN content features
        spk:      [1, 256]     speaker embedding
        k_mem/v_mem: [1, 32, 256] precomputed GTM timbre memory
        kv:       [L, 2, 1, H, C, Dh] KV cache
        kv_valid: int64 scalar, number of real frames at the end of the cache
        returns   mel [1, chunk, 80] (model range, before the vocoder's (m+1)/2), kv_out
        """
        dit = self.dit
        total = self.cache_frames + self.window
        idx = torch.arange(total, device=x.device)
        pad = idx < (self.cache_frames - kv_valid)
        allowed = self.chunk_mask & ~pad[None, None, :]
        bias = torch.where(allowed, 0.0, float("-inf")).unsqueeze(1)  # [L, 1, W, total]

        timbre = dit.temporal_timbre(cond, k_mem, v_mem)
        spks = spk.unsqueeze(1).expand(-1, cond.shape[1], -1)
        dt = 1.0 / ODE_STEPS
        new_kv = kv
        for s in range(ODE_STEPS):
            t_emb = self.t_embs[s]
            h = dit.input_embed(x, timbre, spks)
            layer_kv = []
            for i, block in enumerate(dit.transformer_blocks):
                norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(h, emb=t_emb)
                a, nk, nv = self._attention(block, norm, i, kv[i, 0], kv[i, 1], bias)
                h = h + gate_msa.unsqueeze(1) * a
                n2 = block.ff_norm(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                h = h + gate_mlp.unsqueeze(1) * block.ff(n2)
                layer_kv.append(torch.stack([nk, nv]))
            u = dit.proj_out(dit.norm_out(h, t_emb))
            x = x - dt * u
            new_kv = torch.stack(layer_kv)
        return x[:, :self.chunk, :], new_kv


class GTM(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.gtm = dit.gtm

    def forward(self, spk):
        return self.gtm(spk)


class VocosSpec(nn.Module):
    """Vocos backbone + head up to the complex spectrum. Input is the vocoder-range mel ((m+1)/2)."""

    def __init__(self, vocos):
        super().__init__()
        self.backbone = vocos.backbone
        self.out = vocos.head.out

    def forward(self, mel):
        x = self.backbone(mel, None)
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=100.0)
        return mag * torch.cos(p), mag * torch.sin(p)


class SpeakerEmbed(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, wav):
        return self.enc(wav)


def istft_center(real: torch.Tensor, imag: torch.Tensor, n_fft: int = 640, hop: int = 160) -> torch.Tensor:
    """torch.istft(center=True, hann periodic) — the exact op the engine reimplements."""
    window = torch.hann_window(n_fft, periodic=True)
    return torch.istft(torch.complex(real, imag), n_fft, hop, n_fft, window, center=True)

