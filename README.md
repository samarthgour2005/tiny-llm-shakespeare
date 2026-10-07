# Tiny GPT — Pretraining a Small Language Model from Scratch

A ~10.7M parameter GPT-style decoder-only transformer, **implemented and pretrained
from scratch in PyTorch** — including a custom tokenizer, model architecture, and
training pipeline — on the Tiny Shakespeare corpus.

No pretrained weights, no Hugging Face `transformers` model classes: every
component (tokenizer, attention, transformer block, training loop, sampler) is
written from first principles to demonstrate a working understanding of how
LLMs are built, not just how to call one.

## Why this project

Most "LLM projects" are really API wrappers or fine-tunes of an existing model.
This one goes one level deeper: it builds the thing itself, end to end —
tokenization, self-attention, positional embeddings, the training loop, and
autoregressive sampling — so it demonstrates architecture-level understanding
rather than tool usage.

## Architecture

- Decoder-only transformer (GPT-2 style), pre-norm residual blocks
- Custom multi-head causal self-attention (explicit QKV projection, scaled
  dot-product attention, causal mask), no external attention library
- 4x-expansion MLP with GELU
- Weight-tied token embedding / output projection
- Custom character-level tokenizer (built directly from the corpus vocabulary)

**Default config** (`model.py: GPTConfig`):

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| Attention heads | 6 |
| Embedding dim | 384 |
| Context length | 256 |
| Dropout | 0.2 |
| Parameters | **~10.7M** |

## Repo structure

```
tiny-llm-shakespeare/
├── data/
│   └── prepare.py       # downloads corpus, builds tokenizer, writes train/val .bin files
├── model.py              # GPT architecture (attention, MLP, block, model) from scratch
├── train.py               # training loop: AdamW, cosine LR schedule, grad clipping, checkpointing
├── sample.py              # autoregressive text generation from a checkpoint
├── colab_train.ipynb      # one-click training notebook for a free Colab T4 GPU
├── requirements.txt
└── README.md
```

## Quickstart

### Option A: Google Colab (recommended, free T4 GPU)
Open `colab_train.ipynb` in Colab, set the runtime to a T4 GPU, and run all cells.

### Option B: Local / your own GPU
```bash
pip install -r requirements.txt

# 1. Download + tokenize the corpus
python data/prepare.py

# 2. Train (~15-25 min on a T4, longer on CPU)
python train.py

# 3. Generate text
python sample.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8 --top_k 50
```

## Results

Trained for 5,000 iterations on a single T4 GPU:

| Metric | Value |
|---|---|
| Final train loss | *fill in after your run* |
| Final val loss | *fill in after your run* |
| Training time | *fill in after your run* |

Sample generation (temperature=0.8, top_k=50):
```
<paste a sample output here after training>
```

> Note: with a character-level tokenizer and this model size, output is
> Shakespeare-*flavored* (correct structure, archaic vocabulary, dialogue
> formatting) rather than fully coherent — this is expected at this scale and
> is consistent with results from comparable small nanoGPT-style models.

## What this demonstrates

- **Transformer architecture from first principles**: multi-head causal
  self-attention implemented explicitly (not via `nn.MultiheadAttention`),
  positional/token embeddings, residual + layernorm placement, weight tying
- **Tokenization**: building a vocabulary and encoder/decoder directly from a
  corpus rather than using a pretrained tokenizer
- **Training infrastructure**: learning rate warmup + cosine decay, gradient
  clipping, periodic validation, checkpointing on best validation loss
- **Autoregressive generation**: temperature and top-k sampling implemented
  from scratch

## Possible extensions

- Swap the character tokenizer for a byte-pair encoding (BPE) tokenizer
- Train on a different focused corpus (a specific author, a codebase, a
  domain-specific text collection)
- Scale up (more layers/heads/embedding dim) and compare validation loss
  and sample quality
- Add KV-caching to speed up generation
- Distill this model into an even smaller one, or use it as the "student" in
  a distillation experiment

## Credits

Architecture and training approach are in the spirit of [Andrej Karpathy's
nanoGPT](https://github.com/karpathy/nanoGPT); all code in this repo is an
independent implementation. Corpus: [Tiny
Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
