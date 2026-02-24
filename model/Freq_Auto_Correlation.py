import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math


class AutoCorrelation(nn.Module):
    def __init__(self, factor=3, dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def time_delay_agg_full(self, values, corr):

        batch, head, channel, length = values.shape
        top_k = max(1, int(self.factor * math.log(length)))

        weights, delay = torch.topk(corr, top_k, dim=-1)  # [B, H, E, top_k]
        weights = torch.softmax(weights, dim=-1)

        tmp_values = values.repeat(1, 1, 1, 2)

        init_index = torch.arange(length, device=values.device).view(1, 1, 1, -1)
        init_index = init_index.expand(batch, head, channel, -1)

        delays_agg = torch.zeros_like(values)
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg += pattern * weights[..., i].unsqueeze(-1)

        return delays_agg

    def time_delay_agg_inference(self, values, corr):

        batch, head, channel, length = values.shape
        top_k = max(1, int(self.factor * math.log(length)))

        mean_corr = corr.mean(dim=2)  # [B, H, L]
        weights, delay = torch.topk(mean_corr, top_k, dim=-1)  # [B, H, top_k]
        weights = torch.softmax(weights, dim=-1)

        tmp_values = values.repeat(1, 1, 1, 2)

        init_index = torch.arange(length, device=values.device).view(1, 1, 1, -1)
        init_index = init_index.expand(batch, head, channel, -1)

        delays_agg = torch.zeros_like(values)
        for i in range(top_k):
            tmp_delay = init_index + delay[:, :, i].view(batch, head, 1, 1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg += pattern * weights[:, :, i].view(batch, head, 1, 1)

        return delays_agg

    def time_delay_agg_training(self, values, corr):

        batch, head, channel, length = values.shape
        top_k = max(1, int(self.factor * math.log(length)))

        mean_corr = corr.mean(dim=(1, 2))  # [B, L]
        _, index = torch.topk(mean_corr.mean(dim=0), top_k)  # [top_k]
        weights = torch.softmax(mean_corr[:, index], dim=-1)  # [B, top_k]
        delays_agg = torch.zeros_like(values)

        for i in range(top_k):
            pattern = torch.roll(values, shifts=-int(index[i]), dims=-1)
            delays_agg += pattern * weights[:, i].view(batch, 1, 1, 1)

        return delays_agg

    def forward(self, qk, values):

        B, L, H, E = qk.shape
        values = values[:, :L, :, :]

        qk_fft = torch.fft.rfft(qk.permute(0, 2, 3, 1), dim=-1)  # [B, H, E, L//2+1]
        corr = torch.fft.irfft(qk_fft * torch.conj(qk_fft), n=L, dim=-1)  # [B, H, E, L]

        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1), corr).permute(0, 3, 1, 2)

        return (V, corr.permute(0, 3, 1, 2)) if self.output_attention else (V, None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor):
        super().__init__()

        self.qk_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.correlation = AutoCorrelation(factor)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, x):

        B, L, _ = x.shape


        qk = self.qk_proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, E]
        values = self.value_proj(x).view(B, L, self.n_heads, -1)

        out, attn = self.correlation(qk, values)
        return self.out_proj(out.reshape(B, L, -1)), attn