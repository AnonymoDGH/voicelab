"""Streaming MeanVC2 on ONNX Runtime with numpy-only DSP.

This is the executable specification of the Rust engine: same graphs, same caches, same
fbank / interpolation / iSTFT / crossfade arithmetic. Golden fixtures are produced from it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

SR = 16000
CHUNK = 2560
FRAME_LEN = 400
FRAME_SHIFT = 160
PADDED_LEN = 512
N_MELS = 80
VC_FRAMES_PER_CHUNK = 16
VOCODER_OVERLAP = 2
UPSAMPLE = 160


# ---------------------------------------------------------------------------
# Kaldi fbank (torchaudio.compliance.kaldi.fbank with the MeanVC2 parameters)
# ---------------------------------------------------------------------------

def mel_banks(num_bins=N_MELS, padded_len=PADDED_LEN, sr=SR, low=20.0, high=0.0) -> np.ndarray:
    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    nyquist = 0.5 * sr
    high = nyquist + high if high <= 0 else high
    num_fft_bins = padded_len // 2
    fft_bin_width = sr / padded_len
    low_mel, high_mel = mel(low), mel(high)
    delta = (high_mel - low_mel) / (num_bins + 1)
    b = np.arange(num_bins, dtype=np.float64)[:, None]
    left = low_mel + b * delta
    center = low_mel + (b + 1.0) * delta
    right = low_mel + (b + 2.0) * delta
    m = mel(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))[None, :]
    up = (m - left) / (center - left)
    down = (right - m) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)  # [80, 257]


POVEY = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))) ** 0.85
BANKS = mel_banks()
EPS = np.finfo(np.float32).eps


def fbank(samples: np.ndarray) -> np.ndarray:
    """samples in [-1, 1] float32 -> [frames, 80] log-mel (Kaldi, snip_edges=True)."""
    x = samples.astype(np.float64) * 32768.0
    n = 1 + (len(x) - FRAME_LEN) // FRAME_SHIFT
    if n <= 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = (frames - 0.97 * prev) * POVEY
    spec = np.fft.rfft(frames, n=PADDED_LEN, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(np.maximum(power @ BANKS.T, EPS)).astype(np.float32)


def interp_linear_align_corners(x: np.ndarray, size: int) -> np.ndarray:
    """F.interpolate(mode='linear', align_corners=True) over axis 0. x: [L, C]."""
    L = x.shape[0]
    pos = np.arange(size, dtype=np.float64) * (L - 1) / (size - 1)
    i0 = np.floor(pos).astype(np.int64).clip(0, L - 1)
    i1 = np.minimum(i0 + 1, L - 1)
    w = (pos - i0).astype(np.float32)[:, None]
    return (x[i0] * (1 - w) + x[i1] * w).astype(np.float32)


def istft_center(real: np.ndarray, imag: np.ndarray, n_fft=640, hop=160) -> np.ndarray:
    """torch.istft(center=True, hann periodic window). real/imag: [bins, T]."""
    T = real.shape[1]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = np.fft.irfft(real + 1j * imag, n=n_fft, axis=0) * window[:, None]  # [n_fft, T]
    total = n_fft + hop * (T - 1)
    y = np.zeros(total)
    env = np.zeros(total)
    for t in range(T):
        y[t * hop:t * hop + n_fft] += frames[:, t]
        env[t * hop:t * hop + n_fft] += window ** 2
    pad = n_fft // 2
    return (y[pad:total - pad] / env[pad:total - pad]).astype(np.float32)


# ---------------------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------------------

def session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class OnnxVC:
    def __init__(self, models: Path, variant: str, threads: int = 1, rng: np.random.Generator | None = None,
                 int8: tuple[str, ...] = ()):
        manifest = json.loads((models / "manifest.json").read_text())
        self.cfg = cfg = manifest["variants"][variant]

        def graph(component: str, file: str) -> ort.InferenceSession:
            if component in int8:
                file = file.replace(".onnx", ".int8.onnx")
            return session(models / file, threads)

        self.asr = graph("asr", cfg["asr"])
        self.dit = graph("dit", cfg["dit"])
        self.gtm = graph("gtm", cfg["gtm"])
        self.vocos = graph("vocos", manifest["vocoder"]["file"])
        self.chunk = cfg["chunk_size"]
        self.block = cfg["block_size"]
        self.window = self.chunk + self.block
        self.rng = rng or np.random.default_rng(0)
        self.noise_fn: Callable[[int], np.ndarray] = lambda t: self.rng.standard_normal((1, t, N_MELS), dtype=np.float32)
        self.trace: list[dict] | None = None
        self.down = np.linspace(1, 0, UPSAMPLE, dtype=np.float32)
        self.up = np.linspace(0, 1, UPSAMPLE, dtype=np.float32)
        self.spk = None
        self.reset()

    def set_speaker(self, spk: np.ndarray) -> None:
        self.spk = spk.reshape(1, 256).astype(np.float32)
        self.k_mem, self.v_mem = self.gtm.run(None, {"spk": self.spk})

    def reset(self) -> None:
        c = self.cfg
        self.samples_cache = np.zeros(0, dtype=np.float32)
        self.frame_cache = np.zeros((0, N_MELS), dtype=np.float32)
        self.enc_cache = None
        self.asr_offset = c["asr_offset_init"]
        self.att_cache = np.zeros((6, 4, c["asr_cache"], 128), dtype=np.float32)
        self.cnn_cache = np.zeros((6, 1, 256, 8), dtype=np.float32)
        self.kv = np.zeros(c["kv_shape"], dtype=np.float32)
        self.kv_valid = 0
        self.vc_offset = 0
        self.noise_cache = None
        self.vocoder_cache = None
        self.last_wav = None
        self.bn_buffer = np.zeros((0, 256), dtype=np.float32)

    def _log(self, **kw) -> None:
        if self.trace is not None:
            self.trace.append({k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in kw.items()})

    def encode_chunk(self, samples: np.ndarray) -> np.ndarray | None:
        c = self.cfg
        padded = np.concatenate([self.samples_cache, samples]) if self.samples_cache.size else samples
        if len(padded) < FRAME_LEN:
            self.samples_cache = padded
            return None
        fb = fbank(padded)
        self._log(kind="fbank", samples=padded, fbank=fb)
        self.samples_cache = padded[FRAME_SHIFT * fb.shape[0]:]
        fb = np.concatenate([self.frame_cache, fb])
        if self.asr_offset >= 4000:
            self.asr_offset = max(c["asr_cache"], self.asr_offset - 4000)
        bns = []
        i = 0
        while i + c["bn_window"] <= fb.shape[0]:
            feeds = {
                "fbank": fb[None, i:i + c["bn_window"]],
                "offset": np.array(self.asr_offset, dtype=np.int64),
                "required_cache_size": np.array(c["asr_cache"], dtype=np.int64),
                "att_cache": self.att_cache,
                "cnn_cache": self.cnn_cache,
            }
            bn, self.att_cache, self.cnn_cache = self.asr.run(None, feeds)
            self._log(kind="asr", offset=self.asr_offset, bn=bn[0])
            bns.append(bn[0])
            self.asr_offset += c["asr_offset_step"]
            i += c["bn_stride"]
        self.frame_cache = fb[i:]
        if not bns:
            return None
        bn = np.concatenate(bns)
        if self.enc_cache is not None:
            bn = np.concatenate([self.enc_cache, bn])
        self.enc_cache = bn[-1:]
        if bn.shape[0] < 2:
            return None
        return interp_linear_align_corners(bn, VC_FRAMES_PER_CHUNK + 1)[1:]

    def vc_step(self, cond: np.ndarray) -> np.ndarray:
        x = self.noise_fn(cond.shape[0]).copy()
        if self.noise_cache is not None:
            x[:, :self.block] = self.noise_cache
        self.noise_cache = x[:, -self.block:].copy()
        if self.vc_offset >= 4000:
            self.vc_offset = 0
            self.kv[:] = 0
            self.kv_valid = 0
        mel, self.kv = self.dit.run(None, {
            "x": x, "cond": cond[None], "spk": self.spk, "k_mem": self.k_mem, "v_mem": self.v_mem,
            "kv": self.kv, "kv_valid": np.array(self.kv_valid, dtype=np.int64),
        })
        self.kv_valid = min(self.kv_valid + self.chunk, self.cfg["cache_frames"])
        self.vc_offset += self.chunk
        self._log(kind="dit", x0=x[0], cond=cond, mel=mel[0])
        return mel[0]  # [chunk, 80]

    def decode_mel(self, mel: np.ndarray) -> np.ndarray:
        if self.vocoder_cache is not None:
            mel = np.concatenate([self.vocoder_cache, mel])
        self.vocoder_cache = mel[-VOCODER_OVERLAP:]
        voc_in = ((mel + 1.0) / 2.0).T[None].astype(np.float32)  # [1, 80, T]
        real, imag = self.vocos.run(None, {"mel": voc_in})
        wav = istft_center(real[0], imag[0])
        self._log(kind="vocoder", wav=wav)
        if self.last_wav is not None:
            front = wav[:UPSAMPLE]
            out = np.concatenate([self.last_wav * self.down + front * self.up, wav[UPSAMPLE:-UPSAMPLE]])
        else:
            out = wav[:-UPSAMPLE]
        self.last_wav = wav[-UPSAMPLE:]
        return out

    def process_chunk(self, samples: np.ndarray) -> np.ndarray | None:
        bn = self.encode_chunk(samples)
        if bn is None:
            return None
        self.bn_buffer = np.concatenate([self.bn_buffer, bn])
        parts = []
        for _ in range(4):
            if self.bn_buffer.shape[0] < self.window:
                break
            cond = self.bn_buffer[:self.window]
            self.bn_buffer = self.bn_buffer[self.chunk:]
            parts.append(self.decode_mel(self.vc_step(cond)))
        return np.concatenate(parts) if parts else None

    def process_stream(self, wav: np.ndarray) -> np.ndarray:
        out = []
        for pos in range(0, len(wav), CHUNK):
            o = self.process_chunk(wav[pos:pos + CHUNK].astype(np.float32))
            if o is not None:
                out.append(o)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def speaker_embedding(models: Path, wav16k: np.ndarray, file: str = "spk_encoder.onnx", threads: int = 4) -> np.ndarray:
    sess = session(models / file, threads)
    return sess.run(None, {"wav": wav16k[None].astype(np.float32)})[0][0]

