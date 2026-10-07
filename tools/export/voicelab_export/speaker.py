"""WavLM-Large + ECAPA-TDNN speaker encoder (MeanVC2 `runtime/src/speaker.py`) without s3prl.

The fine-tuned checkpoint (`wavlm_large_finetune.pth`) carries every weight; the WavLM
backbone is built from its known Large config. `hidden_states` replicates s3prl's
UpstreamExpert hooks: the input of each of the 24 encoder layers plus the encoder output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vendor.wavlm.WavLM import WavLM, WavLMConfig

WAVLM_LARGE_CFG = {
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "activation_fn": "gelu",
    "layer_norm_first": True,
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_bias": False,
    "feature_grad_mult": 0.0,
    "normalize": True,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.0,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
}


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias) for _ in range(self.nums)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = None
        for i in range(self.nums):
            sp = spx[i] if sp is None else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class WavLMHiddenStates(nn.Module):
    """s3prl UpstreamExpert equivalent: returns the 25 hidden states of WavLM-Large."""

    def __init__(self):
        super().__init__()
        self.model = WavLM(WavLMConfig(WAVLM_LARGE_CFG))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [1, N] at 16 kHz -> [25, 1, T, 1024]
        if self.model.cfg.normalize:
            wav = F.layer_norm(wav, wav.shape[-1:])
        m = self.model
        features = m.feature_extractor(wav).transpose(1, 2)
        features = m.layer_norm(features)
        if m.post_extract_proj is not None:
            features = m.post_extract_proj(features)
        enc = m.encoder
        x = features + enc.pos_conv(features.transpose(1, 2)).transpose(1, 2)
        x = x.transpose(0, 1)
        states = []
        pos_bias = None
        for layer in enc.layers:
            states.append(x.transpose(0, 1))
            x, _, pos_bias = layer(x, self_attn_padding_mask=None, need_weights=False, pos_bias=pos_bias)
        x = x.transpose(0, 1)
        if enc.layer_norm_first:
            x = enc.layer_norm(x)
        states.append(x)
        return torch.stack(states, dim=0)


class SpeakerEncoder(nn.Module):
    """ECAPA_TDNN(feat_type='wavlm_large', feat_dim=1024, channels=512, emb_dim=256)."""

    def __init__(self, feat_dim=1024, channels=512, emb_dim=256):
        super().__init__()
        self.feature_extract = WavLMHiddenStates()
        self.feature_weight = nn.Parameter(torch.zeros(25))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        ch = [channels] * 4 + [1536]
        self.layer1 = Conv1dReluBn(feat_dim, ch[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(ch[0], ch[1], 3, 1, 2, 2, 8, 128)
        self.layer3 = SE_Res2Block(ch[1], ch[2], 3, 1, 3, 3, 8, 128)
        self.layer4 = SE_Res2Block(ch[2], ch[3], 3, 1, 4, 4, 8, 128)
        self.conv = nn.Conv1d(channels * 3, ch[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(ch[-1], attention_channels=128)
        self.bn = nn.BatchNorm1d(ch[-1] * 2)
        self.linear = nn.Linear(ch[-1] * 2, emb_dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.feature_extract(wav)  # [25, 1, T, 1024]
        w = F.softmax(self.feature_weight, dim=-1).view(-1, 1, 1, 1)
        x = (w * x).sum(dim=0).transpose(1, 2) + 1e-6
        x = self.instance_norm(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        out = self.bn(self.pooling(out))
        return self.linear(out)


def load_speaker_encoder(ckpt_path: str) -> SpeakerEncoder:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    model = SpeakerEncoder()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Pretraining-only heads of WavLM are absent from the forward pass.
    ignorable = ("mask_emb", "label_embs_concat", "final_proj", "project_q", "quantizer", "num_batches_tracked", "loss_calculator")
    missing = [k for k in missing if not any(s in k for s in ignorable)]
    unexpected = [k for k in unexpected if not any(s in k for s in ignorable)]
    if missing or unexpected:
        raise RuntimeError(f"speaker checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.eval()


def load_wav_16k(path: str) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=16000, mono=True)
    return wav.astype(np.float32)
