"""Training loop and evaluation pipeline for MiniGPT."""

import argparse
import os
import random
import time

import torch
import torch.nn as nn

from model import GPTConfig, MiniGPT
from prepare_data import get_dataloaders


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Evaluate model on validation data
@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=20):
    model.eval()

    total_loss = 0.0
    count = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        total_loss += loss.item()
        count += 1

    model.train()

    return total_loss / count if count > 0 else float("inf")


# Train Mini GPT
def train(args):
    set_seed(args.seed)

    device = torch.device(args.device)

    # Load training and validation data
    train_loader, val_loader, tokenizer = get_dataloaders(
        data_path=args.data_path,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    # Initialize model configuration
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_head=args.d_head,
        dropout=args.dropout
    )

    model = MiniGPT(config).to(device)

    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(
        f"MiniGPT initialized on {device} "
        f"({num_params:,} parameters, vocab size {config.vocab_size})"
    )

    # Set optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    model.train()

    step = 0
    start_time = time.time()

    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else args.epochs * len(train_loader)
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        for x, y in train_loader:
            step += 1

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)

            # Calculate loss between predicted and target tokens
            loss = criterion(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1)
            )

            loss.backward()

            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Log training loss
            if step % args.log_interval == 0 or step == 1:
                print(
                    f"epoch {epoch}/{args.epochs} | "
                    f"step {step}/{total_steps} | "
                    f"train loss: {loss.item():.4f}"
                )

            # Evaluate model periodically
            if step % args.eval_interval == 0 or step == total_steps:
                val_loss = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    max_batches=args.eval_batches
                )

                print(f"  --> step {step} | val loss: {val_loss:.4f}")

            # Stop training when maximum steps are reached
            if args.max_steps > 0 and step >= args.max_steps:
                break

        if args.max_steps > 0 and step >= args.max_steps:
            break

    # Calculate training time and final validation loss
    elapsed = time.time() - start_time

    final_val = evaluate(
        model,
        val_loader,
        criterion,
        device,
        max_batches=len(val_loader)
    )

    print(
        f"Finished training in {elapsed:.1f}s. "
        f"Final val loss: {final_val:.4f}"
    )

    # Save model checkpoint and vocabulary
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "final_val_loss": final_val,
        "step": step
    }

    torch.save(checkpoint, args.save_path)
    tokenizer.save_vocab(args.vocab_path)

    print(
        f"Saved checkpoint to {args.save_path} "
        f"and vocab to {args.vocab_path}"
    )


# Parse training arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MiniGPT on TinyStories"
    )

    parser.add_argument("--data_path", default="data/tinystories_sample.txt")
    parser.add_argument("--save_path", default="checkpoints/model.pt")
    parser.add_argument("--vocab_path", default="checkpoints/vocab.json")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_head", type=int, default=16)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=150)
    parser.add_argument("--eval_batches", type=int, default=15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

