import argparse
import random

import torch
import torch.nn.functional as F

from model import GPTConfig, MiniGPT
from prepare_data import SimpleWordTokenizer


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Load trained model and vocabulary
def load_model(
    checkpoint_path="checkpoints/model.pt",
    vocab_path="checkpoints/vocab.json",
    device="cpu"
):
    tokenizer = SimpleWordTokenizer()
    tokenizer.load_vocab(vocab_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])

    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


# Select the next token using temperature sampling
def sample_next_token(logits, temperature):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1).item()

    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# Generate text using incremental decoding and KV caching
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt="Once upon a time",
    max_new_tokens=40,
    temperature=0.7,
    device="cpu"
):
    model.eval()

    prompt = prompt.strip() if prompt else "Once upon a time"

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        token_ids = tokenizer.encode("Once upon a time")

    # Process the complete prompt and initialize the KV cache
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits, cache = model(x)

    next_logits = logits[:, -1, :]

    generated_ids = list(token_ids)

    # Generate tokens one at a time using the cached keys and values
    for _ in range(max_new_tokens):
        if len(generated_ids) >= model.config.max_len:
            break

        next_id = sample_next_token(next_logits, temperature)

        if next_id == tokenizer.eos_id:
            break

        generated_ids.append(next_id)

        if len(generated_ids) >= model.config.max_len:
            break

        # Feed only the latest token and reuse the KV cache
        curr_token = torch.tensor(
            [[next_id]],
            dtype=torch.long,
            device=device
        )

        logits, cache = model(curr_token, cache=cache)
        next_logits = logits[:, -1, :]

    return tokenizer.detokenize(generated_ids)


# Parse generation arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using MiniGPT and KV Cache"
    )

    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        nargs="?",
        const=""
    )

    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--vocab", default="checkpoints/vocab.json")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    model, tokenizer = load_model(
        args.checkpoint,
        args.vocab,
        args.device
    )

    output = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device
    )

    print("\n" + output + "\n")

