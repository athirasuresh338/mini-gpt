import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Model configuration
@dataclass
class GPTConfig:
    vocab_size: int = 8345
    max_len: int = 128
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    d_head: int = 16
    dropout: float = 0.1


# Multi-head causal self-attention with KV caching
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_model == config.n_head * config.d_head

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
        B, T, C = x.shape

        # Project input into query, key and value
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Add previously cached keys and values
        if layer_cache is not None:
            cached_k, cached_v = layer_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v)
        total_len = k.size(2)

        # Calculate scaled attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_head)

        # Apply causal masking
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device)
        k_pos = torch.arange(total_len, device=x.device)

        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Combine values using attention weights
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out, new_cache


# Feedforward network
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return self.dropout(x)


# Transformer block with pre-layer normalization
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x, layer_cache=None, start_pos=0):
        attn_out, new_cache = self.attn(
            self.ln_1(x),
            layer_cache=layer_cache,
            start_pos=start_pos
        )

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        return x, new_cache


# Mini GPT model
class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    # Initialize model weights
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    # Forward pass through the model
    def forward(self, idx, cache=None, start_pos=None):
        B, T = idx.shape

        # Determine the starting position during cached generation
        if start_pos is None:
            if cache is not None:
                start_pos = cache[0][0].size(2)
            else:
                start_pos = 0

        if start_pos + T > self.config.max_len:
            raise ValueError(
                f"Sequence length {start_pos + T} exceeds "
                f"maximum context length {self.config.max_len}"
            )

        # Add token and positional embeddings
        positions = torch.arange(
            start_pos, start_pos + T, device=idx.device
        )

        x = self.token_embedding(idx) + self.pos_embedding(positions)
        x = self.drop(x)

        # Pass through transformer blocks and update KV cache
        new_cache = []

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None

            x, updated_cache = block(
                x,
                layer_cache=layer_cache,
                start_pos=start_pos
            )

            new_cache.append(updated_cache)

        # Generate next-token logits
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_cache

