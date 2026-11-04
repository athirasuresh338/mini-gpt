# Mini GPT from Scratch with KV Caching

A lightweight, decoder-only Transformer implemented from scratch in Python and PyTorch.

The goal of this project was to understand how a Transformer works internally, how it generates text one token at a time, and how **Key-Value (KV) caching** eliminates redundant computation during autoregressive generation.

The model is trained on a sample of 5,000 stories from the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

---

## Project Overview

- **Decoder-Only Transformer**: Built from first principles without high-level library abstractions (no Hugging Face `transformers`), including token embeddings, learned positional embeddings, multi-head causal self-attention, Pre-LayerNorm residuals, and feed-forward networks.
- **Custom Tokenizer**: Implemented a regex-based word-level tokenizer with special tokens for padding (`<PAD>`), unknown words (`<UNK>`), and sequence endings (`<EOS>`).
- **Next-Token Prediction**: Trained on shifted input-target sequence pairs for autoregressive language modeling.
- **KV Caching**: Per-layer key-value caching to store and reuse previous keys and values during incremental decoding.
- **Mathematical Equivalence Verification**: Rigorous unit tests confirming that cached incremental decoding produces logits numerically identical to a full uncached forward pass ($< 10^{-5}$ tolerance).

---

## Model Architecture

```text
Input Token IDs
      |
      v
Token + Positional Embeddings
      |
      v
Transformer Block × 2
├── LayerNorm (Pre-LN)
├── Causal Multi-Head Self-Attention
│   └── KV Cache
├── Residual Connection
├── LayerNorm (Pre-LN)
├── Feed-Forward Network (GELU)
└── Residual Connection
      |
      v
Final LayerNorm
      |
      v
Linear Output Head
      |
      v
Next-Token Logits
```

Each Transformer block utilizes Pre-LayerNorm, causal multi-head self-attention, a feed-forward network, and residual connections. The model contains 2 stacked blocks. 

The KV cache stores computed keys and values at each layer during generation so that subsequent tokens only require computing projections for the single new token.

---

## Model Configuration

| Parameter | Value |
| :--- | :--- |
| **Trainable Parameters** | 1,176,448 (~1.18 million) |
| **Vocabulary Size** | 8,345 unique tokens |
| **Transformer Blocks (`n_layer`)** | 2 |
| **Attention Heads (`n_head`)** | 4 |
| **Embedding Dimension (`d_model`)** | 64 |
| **Head Dimension (`d_head`)** | 16 |
| **Feed-forward Dimension** | 256 ($4 \times d_{model}$) |
| **Context Length (`max_len`)** | 128 tokens |
| **Dropout** | 0.1 |

> [!NOTE]
> The language modeling head (`lm_head`) uses a separate linear layer without bias; its weights are untied from the token embedding weights.

---

## KV Cache Implementation

During text generation, the model operates in two distinct phases:

### 1. Prefill
The complete prompt sequence is passed through the model. The key ($K$) and value ($V$) tensors generated across each attention block are stored in the per-layer cache.

### 2. Incremental Decoding
Only the single newly generated token is passed through the network on each subsequent step. The incoming token's $K$ and $V$ vectors are concatenated with the cached tensors:

$$\text{Cache}_{K} \leftarrow [\text{Cache}_{K}, K_{\text{new}}], \quad \text{Cache}_{V} \leftarrow [\text{Cache}_{V}, V_{\text{new}}]$$

### Computational Complexity Comparison
- **Without KV Cache**: To predict token $t+1$, the model passes all tokens $1 \dots t$ through every layer. Cumulative attention computation scales as approximately **$\mathcal{O}(N^3)$**.
- **With KV Cache**: Each step evaluates attention for only the single active query against cached keys/values, scaling attention computation as **$\mathcal{O}(N^2)$**.

---

## KV Cache Verification

A dedicated unit test compares logits produced by a standard full-sequence forward pass with those produced by incremental decoding using the KV cache.

**Test Configuration:**
- Sequence length: 10 tokens
- Initial prompt length: 3 tokens
- Incremental decoding: Remaining 7 tokens processed one-by-one with `layer_cache`
- Dropout: Disabled during evaluation mode (`eval()`)
- Numerical tolerance threshold: $1.00 \times 10^{-5}$

**Result:**
- **Maximum Absolute Difference**: $\approx 9.69 \times 10^{-8}$ (well within machine precision $< 10^{-5}$)

Run the test suite using:
```bash
python test_cache.py
```

---

## Text Generation Examples

The model generates text autoregressively using temperature sampling:

### Example 1
**Prompt:**
```text
Once upon a time
```
**Generated Text:**
```text
Once upon a time, there was a little girl who lived in the trees. One day, she had a big tree. She was a little girl named Lily and so happy that her mom went to find a big, but he had a big car. She was a
```

### Example 2
**Prompt:**
```text
The little dog
```
**Generated Text:**
```text
The little dog. They had a great time to help. They had a big lake. They were very scared and beautiful car. But it was very happy. He was very happy. The little girl was so brave. He had a big
```

---

## Limitations

- **Model Capacity**: At ~1.18 million parameters, the model is designed for educational exploration of Transformer mechanics rather than complex reasoning or production text generation.
- **Dataset Scale**: Trained on 5,000 short stories from TinyStories for 5 epochs.
- **Tokenizer**: Custom word-level tokenizer with fixed vocabulary; unseen words map to `<UNK>`.
- **Coherence**: Small model size leads to repetition and grammatical drift over long contexts.

---

## Installation and Usage

### 1. Clone the repository
```bash
git clone https://github.com/athirasuresh338/gpt.git
cd gpt
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
```bash
python prepare_data.py
```
Streams 5,000 stories from `TinyStories`, builds `checkpoints/vocab.json`, and prepares train/validation splits.

### 4. Run unit tests and verify the KV cache
```bash
python test_cache.py
```

### 5. Train the model
```bash
python train.py
```

### 6. Generate text
```bash
python generate.py "Once upon a time"
```
You can also adjust generation parameters:
```bash
python generate.py "Once upon a time" --max_new_tokens 50 --temperature 0.7
```

