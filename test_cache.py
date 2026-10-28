import torch
import torch.nn as nn
from model import GPTConfig, MiniGPT


def test_output_shape():
    """
    Check 1: Confirm model output shape is [batch_size, sequence_length, vocab_size].
    """
    config = GPTConfig(vocab_size=1000, max_len=128, d_model=64, n_layer=2, n_head=4, d_head=16)
    model = MiniGPT(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        logits, cache = model(x)

    assert logits.shape == (2, 16, config.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert len(cache) == config.n_layer, f"Expected {config.n_layer} cache entries"
    print("Check 1 passed: model output shape is correct.")


def test_training_step():
    """
    Check 2: Confirm a single training step runs forward + backward and computes a finite loss.
    """
    config = GPTConfig(vocab_size=1000, max_len=128, d_model=64, n_layer=2, n_head=4, d_head=16)
    model = MiniGPT(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    optimizer.zero_grad()
    logits, _ = model(inputs)
    loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"Check 2 passed: training step produces valid loss ({loss.item():.4f}).")


def test_kv_cache():
    """
    Check 3: Verify KV cache numerical equivalence against full uncached sequence forward pass.
    """
    torch.manual_seed(42)
    # dropout=0.0 and eval mode for deterministic equivalence
    config = GPTConfig(vocab_size=500, max_len=128, d_model=64, n_layer=2, n_head=4, d_head=16, dropout=0.0)
    model = MiniGPT(config)
    model.eval()

    total_len = 10
    prompt_len = 3
    seq = torch.randint(0, config.vocab_size, (1, total_len))

    with torch.no_grad():
        # 1. Full sequence pass
        uncached_logits, _ = model(seq)

        # 2. Incremental cached pass: prompt prefill + single-token decoding
        prompt = seq[:, :prompt_len]
        prompt_logits, cache = model(prompt)
        cached_logits = [prompt_logits[:, i : i + 1] for i in range(prompt_len)]

        for step in range(prompt_len, total_len):
            token = seq[:, step : step + 1]
            step_logits, cache = model(token, cache=cache)
            cached_logits.append(step_logits)

        cached_logits = torch.cat(cached_logits, dim=1)

    max_diff = torch.max(torch.abs(uncached_logits - cached_logits)).item()
    print(f"Check 3: max difference between cached and uncached logits = {max_diff:.8e}")

    tolerance = 1e-5
    assert max_diff < tolerance, f"Cache logits differ by {max_diff}, exceeding tolerance {tolerance}"
    print("Check 3 passed: KV cache is mathematically equivalent to full sequence attention.")


if __name__ == "__main__":
    print("Running basic tests...")
    test_output_shape()
    test_training_step()
    test_kv_cache()
    print("All checks passed.")
