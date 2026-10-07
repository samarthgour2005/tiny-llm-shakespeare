"""
Pretraining loop for the tiny GPT model.

Run:
    python train.py

On a Colab T4 GPU this trains a ~10M parameter model on Tiny Shakespeare
in roughly 15-25 minutes and reaches a validation loss around 1.4-1.5
(character-level), which produces fairly coherent Shakespeare-flavored
text (see samples/).
"""
import math
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Config — tuned for a ~10M param model on a single Colab T4 GPU
# ---------------------------------------------------------------------------
out_dir = "checkpoints"
data_dir = "data"

block_size = 256
batch_size = 64
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

max_iters = 5000
eval_interval = 250
eval_iters = 200
log_interval = 50

learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 200
lr_decay_iters = max_iters
weight_decay = 0.1
grad_clip = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1337
# ---------------------------------------------------------------------------

torch.manual_seed(seed)
os.makedirs(out_dir, exist_ok=True)


def load_data():
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(
            "train.bin/val.bin not found. Run `python data/prepare.py` first."
        )
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    return train_data, val_data


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # cosine decay to min_lr
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    train_data, val_data = load_data()

    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPT(config).to(device)
    print(f"Model params: {model.num_params():,}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99)
    )

    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(max_iters + 1):
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "iter": it,
                        "val_loss": best_val_loss,
                    },
                    os.path.join(out_dir, "ckpt.pt"),
                )
                print(f"  -> saved checkpoint (val loss {best_val_loss:.4f})")

        X, Y = get_batch("train", train_data, val_data)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if it % log_interval == 0:
            dt = time.time() - t0
            print(f"iter {it}: loss {loss.item():.4f}, lr {lr:.2e}, {dt:.1f}s elapsed")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
