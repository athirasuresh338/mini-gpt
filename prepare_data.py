import json
import os
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "data/tinystories_sample.txt"
VOCAB_PATH = "checkpoints/vocab.json"
MAX_CONTEXT = 128


class SimpleWordTokenizer:
    """
    A simple word-level tokenizer that splits text into words and punctuation marks.
    Vocabulary is built strictly from the training stories.
    """
    PAD = "<PAD>"
    UNK = "<UNK>"
    EOS = "<EOS>"

    def __init__(self):
        self.token_to_id = {self.PAD: 0, self.UNK: 1, self.EOS: 2}
        self.id_to_token = {0: self.PAD, 1: self.UNK, 2: self.EOS}

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

    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        for special in [self.PAD, self.UNK, self.EOS]:
            counter.pop(special, None)

        words = sorted([w for w, c in counter.items() if c >= min_freq], key=lambda w: (-counter[w], w))

        idx = len(self.token_to_id)
        for w in words:
            self.token_to_id[w] = idx
            self.id_to_token[idx] = w
            idx += 1

    def encode(self, text):
        return [self.token_to_id.get(tok, self.unk_id) for tok in self.tokenize(text)]

    def decode(self, ids):
        return [self.id_to_token.get(idx, self.UNK) for idx in ids]

    def detokenize(self, tokens_or_ids):
        if tokens_or_ids and isinstance(tokens_or_ids[0], int):
            tokens = self.decode(tokens_or_ids)
        else:
            tokens = tokens_or_ids

        text = ""
        no_lead_space = {".", ",", "!", "?", ";", ":", "'", '"', ")", "]", "}"}
        for tok in tokens:
            if tok in (self.PAD, self.EOS):
                continue
            if not text:
                text = tok
            elif tok in no_lead_space:
                text += tok
            else:
                text += " " + tok
        return text

    def save_vocab(self, filepath=VOCAB_PATH):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_vocab(self, filepath=VOCAB_PATH):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}


class StoryDataset(Dataset):
    """
    Dataset of fixed-length token sequences for next-token prediction.
    Each item returns (input_ids, target_ids) where targets are shifted right by 1 token.
    """
    def __init__(self, chunks, pad_id=0, max_len=MAX_CONTEXT):
        self.chunks = chunks
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        chunk = self.chunks[i]
        # input: tokens[:-1], target: tokens[1:]
        inp = chunk[:-1]
        tgt = chunk[1:]

        pad = self.max_len - len(inp)
        if pad > 0:
            inp = inp + [self.pad_id] * pad
            tgt = tgt + [self.pad_id] * pad
        else:
            inp = inp[: self.max_len]
            tgt = tgt[: self.max_len]

        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def load_stories(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()]


def create_chunks(stories, tokenizer, max_len=MAX_CONTEXT):
    # Tokenize stories, append <EOS>, and slice into chunks of max_len + 1
    chunks = []
    for s in stories:
        tokens = tokenizer.encode(s) + [tokenizer.eos_id]
        for i in range(0, len(tokens), max_len):
            c = tokens[i : i + max_len + 1]
            if len(c) >= 2:
                chunks.append(c)
    return chunks


def get_dataloaders(data_path=DATA_PATH, vocab_path=VOCAB_PATH, batch_size=32, max_len=MAX_CONTEXT, rebuild_vocab=False):
    # 90/10 train-val split by story to prevent data leakage
    stories = load_stories(data_path)
    n_train = int(len(stories) * 0.9)
    train_stories, val_stories = stories[:n_train], stories[n_train:]

    tokenizer = SimpleWordTokenizer()
    if os.path.exists(vocab_path) and not rebuild_vocab:
        tokenizer.load_vocab(vocab_path)
    else:
        tokenizer.build_vocab(train_stories)
        tokenizer.save_vocab(vocab_path)

    train_chunks = create_chunks(train_stories, tokenizer, max_len)
    val_chunks = create_chunks(val_stories, tokenizer, max_len)

    train_dataset = StoryDataset(train_chunks, pad_id=tokenizer.pad_id, max_len=max_len)
    val_dataset = StoryDataset(val_chunks, pad_id=tokenizer.pad_id, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer


def download_tinystories(n=5000, out_path=DATA_PATH):
    # Stream n stories from Hugging Face without downloading the entire dataset
    from datasets import load_dataset
    print(f"Connecting to Hugging Face to stream {n} stories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    stories = []
    for item in ds:
        text = item.get("text", "")
        if isinstance(text, str) and text.strip():
            stories.append(text.strip())
        if len(stories) % 500 == 0 and len(stories) > 0:
            print(f"Retrieved {len(stories)} / {n} stories...")
        if len(stories) >= n:
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in stories:
            f.write(s + "\n<|endoftext|>\n")
    print(f"Saved {len(stories)} stories to {out_path}")


if __name__ == "__main__":
    download_tinystories(5000)
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=4, rebuild_vocab=True)
    x, y = next(iter(train_loader))
    print(f"Batch shapes: inputs={x.shape}, targets={y.shape}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    sample_text = "Once upon a time, Lily found a kitten."
    enc = tokenizer.encode(sample_text)
    print(f"Encoded IDs: {enc}")
    print(f"Detokenized: {tokenizer.detokenize(enc)}")
