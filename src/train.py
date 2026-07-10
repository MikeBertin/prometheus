"""
PROMETHEUS — train.py

Trains the byte-level Llama-style transformer on tiny-Shakespeare, then
exports straight to run.c's binary format.

    .venv/bin/python src/train.py                    # full run
    .venv/bin/python src/train.py --iters 50         # smoke test
    .venv/bin/python src/train.py --out models/x.bin --ckpt models/x.pt

Tokenization is trivial by design: token = byte value + 3 (the byte-token
offset run.c's encode() already uses as its fallback). We insert BOS (id 1)
at every blank-line boundary, so the model learns BOS = "a new speaker block
starts here" — that's also what makes unprompted generation from run.c (which
always starts from BOS) produce sensible openings instead of noise.
"""
import argparse
import math
import time

import torch

from model import Transformer, ModelArgs
from export import legacy_export

# ----------------------------- hyperparameters ------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3       # small model, byte vocab: can run hot
MIN_LR = 1e-4              # cosine decays to this
WARMUP_ITERS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 250
EVAL_ITERS = 20
VAL_FRACTION = 0.05
BOS = 1


def load_data(path: str, device: str):
    """bytes -> token ids (byte + 3), BOS at every blank-line boundary."""
    raw = open(path, "rb").read()
    ids = []
    for para in raw.split(b"\n\n"):
        if not para:
            continue
        ids.append(BOS)
        ids.extend(b + 3 for b in para + b"\n\n")
    data = torch.tensor(ids, dtype=torch.int64)
    n_val = int(len(data) * VAL_FRACTION)
    print(f"data: {len(data):,} tokens ({n_val:,} held out for val)")
    return data[:-n_val].to(device), data[-n_val:].to(device)


def get_batch(data, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (BATCH_SIZE,), device=device)
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, splits, seq_len, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, seq_len, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it, max_iters):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    ratio = (it - WARMUP_ITERS) / max(1, max_iters - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--out", default="models/shakespeare.bin")
    ap.add_argument("--ckpt", default="models/ckpt.pt")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model_args = ModelArgs()  # the Phase-2 config lives in model.py defaults
    model = Transformer(model_args).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params")

    train_data, val_data = load_data(args.data, device)
    splits = {"train": train_data, "val": val_data}

    # AdamW with weight decay on the matrices only (norm gains + embeddings
    # stay undecayed — standard practice, keeps norms free to scale).
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=(0.9, 0.95))

    seq_len = model_args.max_seq_len
    t0 = time.time()
    for it in range(args.iters):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(it, args.iters)

        x, y = get_batch(train_data, seq_len, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == args.iters - 1:
            losses = estimate_loss(model, splits, seq_len, device)
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {dt:6.1f}s")

    # sanity sample from PyTorch before exporting (BOS-primed)
    start = torch.tensor([[BOS]], dtype=torch.int64, device=device)
    sample = model.generate(start, 200, temperature=0.8)[0].tolist()
    text = bytes(t - 3 for t in sample if t >= 3).decode("utf-8", errors="replace")
    print(f"--- pytorch sample ---\n{text}\n----------------------")

    torch.save({"model": model.state_dict(),
                "model_args": model_args.__dict__}, args.ckpt)
    model.cpu()
    legacy_export(model, args.out)


if __name__ == "__main__":
    main()
