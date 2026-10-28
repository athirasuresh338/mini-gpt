import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8345
    max_len: int = 128
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    d_head: int = 16
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking and explicit KV caching.
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model == config.n_head * config.d_head, "d_model must equal n_head * d_head"

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, layer_cache=None, start_pos=0):
        # x: (B, T, d_model)
        B, T, C = x.shape

        # (B, T, d_model) -> (B, n_head, T, d_head)
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Concatenate with cached past keys/values along sequence dimension
        if layer_cache is not None:
            cached_k, cached_v = layer_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v)
        total_len = k.size(2)

        # Scaled dot-product: (B, n_head, T, d_head) @ (B, n_head, d_head, total_len) -> (B, n_head, T, total_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask: query at (start_pos + i) can only attend to keys at j <= (start_pos + i)
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(1)
        k_pos = torch.arange(0, total_len, device=x.device).unsqueeze(0)
        mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # (B, n_head, T, total_len) @ (B, n_head, total_len, d_head) -> (B, n_head, T, d_head)
        out = torch.matmul(weights, v)
        # (B, n_head, T, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.out_proj(out))
        return out, new_cache


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x, layer_cache=None, start_pos=0):
        attn_out, new_cache = self.attn(self.ln_1(x), layer_cache=layer_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, cache=None, start_pos=None):
        # idx: (B, T)
        B, T = idx.shape

        # start_pos tracks position offset when generating incrementally with KV cache
        if start_pos is None:
            if cache is not None and cache[0] is not None:
                start_pos = cache[0][0].size(2)
            else:
                start_pos = 0

        if start_pos + T > self.config.max_len:
            raise ValueError(f"Sequence length {start_pos + T} exceeds max context length {self.config.max_len}")

        device = idx.device
        positions = torch.arange(start_pos, start_pos + T, device=device)

        x = self.drop(self.token_embedding(idx) + self.pos_embedding(positions))

        new_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, updated_layer_cache = block(x, layer_cache=layer_cache, start_pos=start_pos)
            new_cache.append(updated_layer_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, new_cache
