# Mini GPT from Scratch with KV Caching

The goal of this project was to understand how a Transformer works internally, how it generates text one token at a time, and how Key-Value (KV) caching reduces repeated computation during generation.

The model was trained on 5,000 short stories from the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

---

## 1. What I Implemented

- **Decoder-Only Transformer**: Built the model architecture from scratch, including token embeddings, positional embeddings, causal self-attention, residual connections, and feedforward networks.
- **Custom Tokenizer**: Implemented a regex-based word-level tokenizer with special tokens for padding, unknown words, and sequence endings.
- **Next-Token Prediction**: Trained the model to predict the next token based on the preceding sequence.
- **KV Caching**: Implemented per-layer key-value caching to reuse previously computed keys and values during autoregressive generation.
- **Cache Verification**: Compared cached and uncached model outputs to verify that caching preserves the model's predictions.

---

## 2. Model Architecture

The model follows a Pre-LayerNorm, decoder-only Transformer architecture.

```
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

### Model Configuration

| Parameter | Value |
|---|---:|
| Parameters | 1.18 million |
| Vocabulary Size | 8,345 |
| Transformer Blocks | 2 |
| Attention Heads | 4 |
| Embedding Dimension | 64 |
| Head Dimension | 16 |
| Feedforward Dim. | 256 |
| Context Length | 128 tokens |
| Dropout | 0.1 |

The model uses causal self-attention so that each token can attend to itself and preceding tokens, rather than future tokens. Residual connections and Pre-LayerNorm are used within each Transformer block.

---

## 3. How KV Caching Works

During autoregressive generation, a Transformer predicts one token at a time. Without caching, the model repeatedly processes previous tokens to calculate the attention keys and values needed for the next prediction.

KV caching avoids this repeated work by storing the keys and values computed by each attention layer.

The implementation has two stages:
- **Prompt Prefill**: The prompt is processed once, creating the key-value cache for each Transformer layer.
- **Incremental Decoding**: Each newly generated token is processed individually. Its keys and values are added to the existing cache, allowing its query to attend to the cached history.

For cumulative attention computation, this changes the scaling from approximately $O(N^3)$ without caching to $O(N^2)$ with caching during autoregressive generation, where $N$ is the sequence length.

---

## 4. Testing KV Cache Correctness

To check the correctness of the cache implementation, `test_cache.py` compares the output logits from cached and uncached inference on the same 10-token sequence.

**Test procedure:**
- **Uncached**: Process the complete sequence in one forward pass.
- **Cached**: Process the first three tokens as the prompt, then process the remaining seven tokens incrementally using the cache.

### Verification Result

```
Maximum Absolute Logit Difference : 2.38 × 10⁻⁷
Numerical Tolerance               : 1.00 × 10⁻⁵
```

The maximum absolute difference is smaller than the specified tolerance. This indicates that cached and uncached inference produce numerically equivalent logits for the tested sequence.

---

## 5. Text Generation and Model Limitations

The trained model generates text autoregressively, predicting each new token based on the preceding context. The following examples were generated with a temperature of 0.6.

### Example 1
**Prompt:** `Once upon a time`  
> *Once upon a time, there was a little girl who lived in the trees. One day, she had a big tree. She was a little girl named Lily and so happy that her mom went to find a big, but he had a big car. She was a*

### Example 2
**Prompt:** `The little dog`  
> *The little dog. They had a great time to help. They had a big lake. They were very scared and beautiful car. But it was very happy. He was very happy. The little girl was so brave. He had a big*

The generated samples show that the model can learn basic language patterns, but it still has limitations.

These limitations reflect the model's small size (approximately 1.18 million parameters), five training epochs, and basic word-level tokenizer.

---

## 6. Project Structure

```
mini-gpt-kv-cache/
├── data/
│   └── tinystories_sample.txt   # 5,000 TinyStories samples
├── checkpoints/
│   ├── model.pt                 # Saved model weights and config
│   └── vocab.json               # Word-level vocabulary
├── model.py                     # Transformer architecture and KV cache
├── prepare_data.py              # Dataset preparation, tokenizer, and chunking
├── train.py                     # Training loop and evaluation
├── generate.py                  # Autoregressive text generation
├── test_cache.py                # KV cache verification
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 7. How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Stories and Build Vocabulary
```bash
python prepare_data.py
```

### 3. Verify KV Cache Equivalence
```bash
python test_cache.py
```

### 4. Train the Model for 5 Epochs
```bash
python train.py --epochs 5 --batch_size 32
```

### 5. Generate Text with KV Caching
```bash
python generate.py --prompt "Once upon a time" --temperature 0.6
```
