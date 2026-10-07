"""
Prepares the Tiny Shakespeare dataset for training.

This builds a custom character-level tokenizer from scratch (no external
tokenizer library) and encodes the corpus into train/val binary files.

Run:
    python data/prepare.py
"""
import os
import pickle
import urllib.request

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(OUT_DIR, "input.txt")


def download_data():
    if os.path.exists(INPUT_PATH):
        print(f"Found existing corpus at {INPUT_PATH}")
        return
    print(f"Downloading corpus from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, INPUT_PATH)
    print("Download complete.")


def build_tokenizer(text):
    """Custom character-level tokenizer: builds a stoi/itos vocab from the
    unique characters seen in the corpus. This is the simplest possible
    tokenizer, but it is genuinely 'from scratch' and works well for a
    small, focused corpus like Shakespeare."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size


def encode(text, stoi):
    return [stoi[c] for c in text]


def main():
    download_data()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"Corpus length: {len(data):,} characters")

    stoi, itos, vocab_size = build_tokenizer(data)
    print(f"Vocab size: {vocab_size} unique characters")

    # 90/10 train/val split
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_ids = encode(train_data, stoi)
    val_ids = encode(val_data, stoi)
    print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))

    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("Saved train.bin, val.bin, and meta.pkl")


if __name__ == "__main__":
    main()
