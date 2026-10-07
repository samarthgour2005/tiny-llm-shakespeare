"""
Generate text from a trained checkpoint.

Run:
    python sample.py --prompt "ROMEO:" --max_new_tokens 300
"""
import argparse
import pickle

import torch

from model import GPT

CKPT_PATH = "checkpoints/ckpt.pt"
META_PATH = "data/meta.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    # weights_only=False: safe here since this is a checkpoint we trained ourselves
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from iter {checkpoint['iter']} (val loss {checkpoint['val_loss']:.4f})")

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join([itos[i] for i in ids])

    start_ids = encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        y = model.generate(
            x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
        )

    print("-" * 60)
    print(decode(y[0].tolist()))
    print("-" * 60)


if __name__ == "__main__":
    main()
