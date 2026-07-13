"""
PROMETHEUS — prepare_data.py

Tokenize the TinyStories corpus ONCE into a flat uint16 array on disk, so
training just memory-maps it and samples random windows (nanoGPT style).
Tokenizing on the fly would bottleneck the GPU; vocab 4096 < 65536 so each
token fits in a uint16 (2 bytes).

Each story is emitted as: <s> (BOS=1) + its BPE tokens. The model thus learns
BOS = "a new story begins", which is what makes generation from run.c (which
starts at BOS) open like a fresh story.

  python prepare_data.py \
      --merges models/tinystories.merges \
      --train data/tinystories_train_300mb.txt \
      --val   data/tinystories_valid.txt

Writes data/tinystories_{train,val}.bin. Parallelized across stories.
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bpe import Tokenizer, load_merges

_TOK = None  # per-worker global (set in each process)


def _init(merges_path):
    global _TOK
    merges, ranks = load_merges(merges_path)
    _TOK = Tokenizer(merges, ranks)


def _encode_story(story):
    # BOS + tokens; skip empty fragments
    s = story.strip()
    if not s:
        return None
    return np.array([1] + _TOK.encode(s), dtype=np.uint16)


def process(path, out_path, merges_path, limit_mb=0.0):
    text = open(path, encoding="utf-8", errors="replace").read()
    if limit_mb:
        text = text[: int(limit_mb * 1_000_000)]
    stories = re.split(r"<\|endoftext\|>", text)
    # drop a possibly-truncated final story (range-downloaded corpus)
    if stories and not text.rstrip().endswith(("\"", ".", "!", "?")):
        stories = stories[:-1]
    print(f"{path}: {len(stories):,} stories, {len(text)/1e6:.0f} MB")

    chunks = []
    total = 0
    with ProcessPoolExecutor(initializer=_init, initargs=(merges_path,)) as ex:
        for i, arr in enumerate(ex.map(_encode_story, stories, chunksize=256)):
            if arr is not None:
                chunks.append(arr)
                total += len(arr)
            if (i + 1) % 50000 == 0:
                print(f"  {i+1:,}/{len(stories):,} stories, {total/1e6:.1f}M tokens")

    all_tokens = np.concatenate(chunks)
    all_tokens.tofile(out_path)
    print(f"wrote {out_path}: {len(all_tokens):,} tokens "
          f"({all_tokens.nbytes/1e6:.0f} MB), "
          f"~{all_tokens.nbytes/len(''.join(stories).encode()):.2f} bytes/char compression")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("--train", default="data/tinystories_train_300mb.txt")
    ap.add_argument("--val", default="data/tinystories_valid.txt")
    ap.add_argument("--train-out", default="data/tinystories_train.bin")
    ap.add_argument("--val-out", default="data/tinystories_val.bin")
    ap.add_argument("--train-mb", type=float, default=0.0)
    args = ap.parse_args()

    process(args.val, args.val_out, args.merges)
    process(args.train, args.train_out, args.merges, limit_mb=args.train_mb)
