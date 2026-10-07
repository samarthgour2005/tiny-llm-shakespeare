# Tiny GPT — LLM Pretraining from Scratch

A **10.7M parameter** GPT-style transformer built and trained from scratch in PyTorch on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) corpus — no pretrained weights, no Hugging Face model classes.

## What's in here

| File | Description |
|---|---|
| `model.py` | GPT architecture (attention, MLP, block) from scratch |
| `train.py` | Training loop — AdamW, cosine LR, grad clipping, checkpointing |
| `sample.py` | Autoregressive text generation from a saved checkpoint |
| `data/prepare.py` | Downloads corpus, builds char tokenizer, writes `train.bin` / `val.bin` |
| `colab_train.ipynb` | One-click notebook for Google Colab (T4 GPU) with interactive inference playground |
| `requirements.txt` | Dependencies |

## Architecture

- Decoder-only transformer (GPT-2 style)
- Pre-norm residual blocks (LayerNorm before attention + MLP)
- Multi-head causal self-attention — explicit QKV projection, no `nn.MultiheadAttention`
- Weight-tied token embedding and output projection
- Character-level tokenizer built directly from the corpus

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| Heads | 6 |
| Embedding dim | 384 |
| Context length | 256 tokens |
| Dropout | 0.2 |
| **Total params** | **10,725,120** |

## Training results

Trained for **5,000 iterations** on a single **T4 GPU** (~72 minutes):

| Metric | Value |
|---|---|
| Best validation loss | **1.4817** (step 3,500) |
| Final train loss | 1.02 |
| Optimizer | AdamW (`β₁=0.9`, `β₂=0.99`, `wd=0.1`) |
| Learning rate | `3e-4` → `3e-5` (cosine decay, 200-iter warmup) |
| Gradient clipping | `1.0` |

See [`RESULTS.md`](RESULTS.md) for the full loss curve and sample outputs.

## Quickstart

### Option A — Google Colab (easiest, free GPU)

Open `colab_train.ipynb` in Colab, set the runtime to **T4 GPU**, and run all cells in order. The last cell is an interactive inference playground.

### Option B — Local

```bash
pip install -r requirements.txt

# 1. Download + tokenize the corpus
python data/prepare.py

# 2. Train
python train.py

# 3. Generate text
python sample.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8 --top_k 50
```

## Sample output

```
ROMEO:
Shall I speak more, or shall I hear at this?

JULIET:
O, then, if thou shouldst be redeemption
On that the delight of York is dead?

JULIET:
Thou hast thou wound'st, for thy windows behind
Thy love's rich letter us dead to one,
And thy father s
```

*(temperature=0.8, top_k=50, trained to step 5,000)*

## Credits

Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT); all code is an independent implementation written for clarity.
