"""
PROMETHEUS — train.py

Trains the Llama-style transformer and exports straight to run.c's format.
Two data modes:

  byte mode (Phase 2, tiny-Shakespeare):
      python train.py --data data/input.txt --iters 2000
      token = byte+3, BOS at blank lines, vocab 259.

  token mode (Phase 5, TinyStories + BPE):
      python train.py --train-bin data/tinystories_train.bin \
          --val-bin data/tinystories_val.bin --merges models/tinystories.merges \
          --dim 288 --layers 6 --heads 6 --hidden 768 --vocab 4096 --iters 6000
      reads a pre-tokenized uint16 memmap (see prepare_data.py); stories are
      already BOS-delimited, so we sample random windows straight from it.

Either way the model config is saved into the checkpoint, so export.py can
reconstruct any size without being told the dims.
"""
import argparse
import math
import time

import numpy as np
import torch

from model import Transformer, ModelArgs
from export import legacy_export

# ----------------------------- fixed hyperparameters ------------------------
MIN_LR_FRAC = 0.1          # cosine floor = LR * this
WARMUP_ITERS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 250
EVAL_ITERS = 20
VAL_FRACTION = 0.05        # byte mode only (token mode has a separate val file)
BOS = 1


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_bytes(path):
    """byte mode: text -> token ids (byte+3), BOS at blank-line boundaries."""
    raw = open(path, "rb").read()
    ids = []
    for para in raw.split(b"\n\n"):
        if not para:
            continue
        ids.append(BOS)
        ids.extend(b + 3 for b in para + b"\n\n")
    data = np.array(ids, dtype=np.uint16)
    n_val = int(len(data) * VAL_FRACTION)
    return data[:-n_val], data[-n_val:]


def load_bin(path):
    """token mode: memory-map a pre-tokenized uint16 array."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, seq_len, batch_size, device):
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i + seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + seq_len].astype(np.int64) for i in ix])
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == "mps":  # non_blocking pin doesn't help mps; just move
        return x.to(device), y.to(device)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, splits, seq_len, batch_size, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, seq_len, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it, max_iters, lr):
    if it < WARMUP_ITERS:
        return lr * (it + 1) / WARMUP_ITERS
    ratio = (it - WARMUP_ITERS) / max(1, max_iters - WARMUP_ITERS)
    return lr * MIN_LR_FRAC + 0.5 * lr * (1 - MIN_LR_FRAC) * (1 + math.cos(math.pi * ratio))


def make_decoder(args):
    """Return a fn: list[int] -> str, for the pre-export sanity sample."""
    if args.merges:
        from bpe import Tokenizer, load_merges
        tok = Tokenizer(*load_merges(args.merges))
        return lambda ids: tok.decode([i for i in ids if i >= BOS])
    return lambda ids: bytes(t - 3 for t in ids if t >= 3).decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    # data
    ap.add_argument("--data", default=None, help="byte mode: text file")
    ap.add_argument("--train-bin", default=None, help="token mode: uint16 train")
    ap.add_argument("--val-bin", default=None, help="token mode: uint16 val")
    ap.add_argument("--merges", default=None, help="BPE merges (for sample decode)")
    # model config (defaults = model.py's Phase-2 Shakespeare values)
    d = ModelArgs()
    ap.add_argument("--dim", type=int, default=d.dim)
    ap.add_argument("--layers", type=int, default=d.n_layers)
    ap.add_argument("--heads", type=int, default=d.n_heads)
    ap.add_argument("--kv-heads", type=int, default=d.n_kv_heads)
    ap.add_argument("--hidden", type=int, default=d.hidden_dim)
    ap.add_argument("--seq-len", type=int, default=d.max_seq_len)
    ap.add_argument("--vocab", type=int, default=d.vocab_size)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    # output
    ap.add_argument("--out", default="models/model.bin")
    ap.add_argument("--ckpt", default="models/ckpt.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model_args = ModelArgs(dim=args.dim, n_layers=args.layers, n_heads=args.heads,
                           n_kv_heads=args.kv_heads, hidden_dim=args.hidden,
                           max_seq_len=args.seq_len, vocab_size=args.vocab,
                           dropout=args.dropout)
    model = Transformer(model_args).to(device)
    print(f"model: dim={args.dim} L={args.layers} H={args.heads} "
          f"hidden={args.hidden} vocab={args.vocab} seq={args.seq_len} "
          f"-> {sum(p.numel() for p in model.parameters()):,} params")

    if args.train_bin:
        train_data, val_data = load_bin(args.train_bin), load_bin(args.val_bin)
    else:
        train_data, val_data = load_bytes(args.data)
    print(f"data: train {len(train_data):,} tok | val {len(val_data):,} tok")
    splits = {"train": train_data, "val": val_data}

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    seq_len, batch = args.seq_len, args.batch
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.iters):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(it, args.iters, args.lr)

        x, y = get_batch(train_data, seq_len, batch, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == args.iters - 1:
            losses = estimate_loss(model, splits, seq_len, batch, device)
            dt = time.time() - t0
            flag = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(),
                            "model_args": model_args.__dict__}, args.ckpt)
                flag = " *"   # checkpointed the val-best model
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {dt:6.1f}s{flag}", flush=True)

    # sanity sample from PyTorch (BOS-primed), using the val-best weights
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    decode = make_decoder(args)
    start = torch.tensor([[BOS]], dtype=torch.int64, device=device)
    ids = model.generate(start, 200, temperature=0.8)[0].tolist()
    print(f"--- pytorch sample (val-best) ---\n{decode(ids)}\n----------------------")

    model.cpu()
    legacy_export(model, args.out)
    print(f"best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
