"""Prepare TinyStories data for training a Mini GPT model."""

import json
import os
import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader


DATA_PATH = "data/tinystories_sample.txt"
VOCAB_PATH = "checkpoints/vocab.json"
MAX_CONTEXT = 128


# Simple word-level tokenizer
class SimpleWordTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    EOS = "<EOS>"

    def __init__(self):
        self.token_to_id = {
            self.PAD: 0,
            self.UNK: 1,
            self.EOS: 2
        }

        self.id_to_token = {
            0: self.PAD,
            1: self.UNK,
            2: self.EOS
        }

    @property
    def pad_id(self):
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self):
        return self.token_to_id[self.UNK]

    @property
    def eos_id(self):
        return self.token_to_id[self.EOS]

    @property
    def pad_token_id(self):
        return self.pad_id

    @property
    def unk_token_id(self):
        return self.unk_id

    @property
    def eos_token_id(self):
        return self.eos_id

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @staticmethod
    def tokenize(text):
        if not text:
            return []

        return re.findall(r"\w+|[^\w\s]", text)

    # Build vocabulary using token frequency
    def build_vocab(self, texts, min_freq=1):
        counter = Counter()

        for text in texts:
            counter.update(self.tokenize(text))

        for special in [self.PAD, self.UNK, self.EOS]:
            counter.pop(special, None)

        words = sorted(
            [word for word, count in counter.items() if count >= min_freq],
            key=lambda word: (-counter[word], word)
        )

        for idx, word in enumerate(words, start=len(self.token_to_id)):
            self.token_to_id[word] = idx
            self.id_to_token[idx] = word

    # Convert text into token IDs
    def encode(self, text):
        return [
            self.token_to_id.get(token, self.unk_id)
            for token in self.tokenize(text)
        ]

    # Convert token IDs back into tokens
    def decode(self, ids):
        return [
            self.id_to_token.get(idx, self.UNK)
            for idx in ids
        ]

    # Convert tokens into readable text
    def detokenize(self, tokens_or_ids):
        if tokens_or_ids and isinstance(tokens_or_ids[0], int):
            tokens = self.decode(tokens_or_ids)
        else:
            tokens = tokens_or_ids

        no_lead_space = {".", ",", "!", "?", ";", ":", "'", '"', ")", "]", "}"}

        text = ""

        for token in tokens:
            if token in (self.PAD, self.EOS):
                continue

            if not text or token in no_lead_space:
                text += token
            else:
                text += " " + token

        return text

    # Save vocabulary to a JSON file
    def save_vocab(self, filepath=VOCAB_PATH):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()}
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Load vocabulary from a JSON file
    def load_vocab(self, filepath=VOCAB_PATH):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self.id_to_token = {
            int(k): v for k, v in data["id_to_token"].items()
        }


# Dataset for next-token prediction
class StoryDataset(Dataset):
    def __init__(self, chunks, pad_id=0, max_len=MAX_CONTEXT):
        self.chunks = chunks
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        chunk = self.chunks[i]

        # Shift tokens to create input and target sequences
        inp = chunk[:-1]
        tgt = chunk[1:]

        # Pad shorter sequences to the maximum length
        pad = self.max_len - len(inp)

        if pad > 0:
            inp = inp + [self.pad_id] * pad
            tgt = tgt + [self.pad_id] * pad
        else:
            inp = inp[:self.max_len]
            tgt = tgt[:self.max_len]

        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long)
        )


# Load stories from the dataset file
def load_stories(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [
            story.strip()
            for story in f.read().split("<|endoftext|>")
            if story.strip()
        ]


# Split stories into fixed-length token chunks
def create_chunks(stories, tokenizer, max_len=MAX_CONTEXT):
    chunks = []

    for story in stories:
        tokens = tokenizer.encode(story) + [tokenizer.eos_id]

        # Keep one overlapping token between consecutive chunks
        for i in range(0, len(tokens), max_len):
            chunk = tokens[i:i + max_len + 1]

            if len(chunk) >= 2:
                chunks.append(chunk)

    return chunks


# Create training and validation DataLoaders
def get_dataloaders(
    data_path=DATA_PATH,
    vocab_path=VOCAB_PATH,
    batch_size=32,
    max_len=MAX_CONTEXT,
    rebuild_vocab=False
):
    stories = load_stories(data_path)

    # Split by story to prevent data leakage
    n_train = int(len(stories) * 0.9)
    train_stories = stories[:n_train]
    val_stories = stories[n_train:]

    tokenizer = SimpleWordTokenizer()

    # Load existing vocabulary or build a new one
    if os.path.exists(vocab_path) and not rebuild_vocab:
        tokenizer.load_vocab(vocab_path)
    else:
        tokenizer.build_vocab(train_stories)
        tokenizer.save_vocab(vocab_path)

    # Create token chunks for both datasets
    train_chunks = create_chunks(train_stories, tokenizer, max_len)
    val_chunks = create_chunks(val_stories, tokenizer, max_len)

    train_dataset = StoryDataset(
        train_chunks,
        pad_id=tokenizer.pad_id,
        max_len=max_len
    )

    val_dataset = StoryDataset(
        val_chunks,
        pad_id=tokenizer.pad_id,
        max_len=max_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, tokenizer


# Download TinyStories from Hugging Face
def download_tinystories(n=5000, out_path=DATA_PATH):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Dataset already exists at {out_path}. Skipping download.")
        return

    from datasets import load_dataset

    print(f"Connecting to Hugging Face to stream {n} stories...")

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True
    )

    stories = []

    for item in dataset:
        text = item.get("text", "")

        if isinstance(text, str) and text.strip():
            stories.append(text.strip())

        if len(stories) > 0 and len(stories) % 500 == 0:
            print(f"Retrieved {len(stories)} / {n} stories...")

        if len(stories) >= n:
            break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for story in stories:
            f.write(story + "\n<|endoftext|>\n")

    print(f"Saved {len(stories)} stories to {out_path}")


# Run data preparation and test the tokenizer
if __name__ == "__main__":
    download_tinystories(5000)

    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=4,
        rebuild_vocab=True
    )

    x, y = next(iter(train_loader))

    print(f"Batch shapes: inputs={x.shape}, targets={y.shape}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    sample_text = "Once upon a time, Lily found a kitten."

    encoded = tokenizer.encode(sample_text)

    print(f"Encoded IDs: {encoded}")
    print(f"Detokenized: {tokenizer.detokenize(encoded)}")
