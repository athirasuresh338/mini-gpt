"""Unit tests for MiniGPT model: shape, training step, and KV cache equivalence."""

import torch
import torch.nn as nn
from model import MiniGPT, GPTConfig


def test_output_shape():
    """Verifies that a random input tensor produces expected logit and cache shapes."""
    config = GPTConfig(
        vocab_size=100,
        max_len=32,
        d_model=32,
        n_layer=2,
        n_head=2,
        d_head=16
    )
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, cache = model(x)

    assert logits.shape == (2, 16, config.vocab_size), (
        f"Expected logits shape {(2, 16, config.vocab_size)}, got {logits.shape}"
    )
    assert len(cache) == config.n_layer, (
        f"Expected {config.n_layer} cache layers, got {len(cache)}"
    )

    for i, (k, v) in enumerate(cache):
        assert k.shape == (2, config.n_head, 16, config.d_head), (
            f"Layer {i} key shape expected {(2, config.n_head, 16, config.d_head)}, got {k.shape}"
        )
        assert v.shape == (2, config.n_head, 16, config.d_head), (
            f"Layer {i} value shape expected {(2, config.n_head, 16, config.d_head)}, got {v.shape}"
        )

    print("[PASS] test_output_shape")


def test_training_step():
    """Verifies forward pass, loss calculation, backward pass, and weight update."""
    config = GPTConfig(
        vocab_size=100,
        max_len=32,
        d_model=32,
        n_layer=2,
        n_head=2,
        d_head=16
    )
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))

    model.train()
    optimizer.zero_grad()

    logits, _ = model(x)
    loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss contains NaN or Inf"

    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient is None for {name}"
        assert not torch.isnan(p.grad).any(), f"Gradient contains NaN for {name}"
        assert not torch.isinf(p.grad).any(), f"Gradient contains Inf for {name}"

    orig_weight = model.token_embedding.weight.clone()
    optimizer.step()

    assert not torch.equal(orig_weight, model.token_embedding.weight), (
        "Token embedding weights were not updated after optimizer step"
    )

    print("[PASS] test_training_step")


def test_kv_cache(tolerance=1e-5):
    """
    Verifies that incremental generation with KV caching produces
    identical logits to a full uncached forward pass within machine precision.
    """
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=100,
        max_len=32,
        d_model=32,
        n_layer=2,
        n_head=2,
        d_head=16,
        dropout=0.0
    )
    model = MiniGPT(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        # 1. Full uncached forward pass on all 10 tokens
        logits_uncached, _ = model(tokens)

        # 2. Cached generation: prefill with 3 tokens, then 7 incremental steps
        prefill_len = 3
        prompt = tokens[:, :prefill_len]
        logits_prefill, cache = model(prompt)

        cached_logits = [logits_prefill]

        for i in range(prefill_len, 10):
            next_token = tokens[:, i : i + 1]
            step_logits, cache = model(next_token, cache=cache)
            cached_logits.append(step_logits)

        logits_cached = torch.cat(cached_logits, dim=1)

        # 3. Verify numerical equivalence
        max_abs_diff = (logits_uncached - logits_cached).abs().max().item()

    print(f"Max absolute difference: {max_abs_diff:.8e} (tolerance: {tolerance:.1e})")
    assert max_abs_diff < tolerance, (
        f"KV cache output diverged: max difference {max_abs_diff} >= {tolerance}"
    )

    print("[PASS] test_kv_cache (Mathematical Equivalence Verified)")


if __name__ == "__main__":
    test_output_shape()
    test_training_step()
    test_kv_cache()
