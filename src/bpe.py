"""
PROMETHEUS — bpe.py

A byte-level BPE tokenizer, trained from scratch, exported into the SAME
tokenizer.bin format run.c already reads. No C changes are needed: run.c's
encode() is a greedy "merge the highest-scored adjacent pair" loop, which is
exactly BPE decoding — so if we write our merges with scores that encode
priority, run.c reproduces our tokenization for free.

Why this matters vs the Phase-2 byte tokenizer: "Once upon a time" was 16
byte-tokens; with a 4096-vocab BPE it's ~4. Shorter sequences = the model's
fixed 256-token context reaches much further into a story, and each step
predicts a chunk of word instead of one letter.

Design, and how it stays consistent with run.c:
  * Base alphabet = 256 raw bytes at ids 3..258 (id = byte + 3). This is the
    exact offset run.c's encode() uses for its byte fallback, so unknown
    bytes map identically on both sides.
  * ids 0/1/2 = <unk>/<s>/</s>. <s> (BOS=1) separates stories in training and
    starts generation in run.c.
  * Merges get ids 259.. and a score = -rank (earliest-learned = highest).
    run.c picks the max-score adjacent merge each step == applying the
    highest-priority BPE rule == standard BPE.
  * Pre-tokenization attaches a leading space to each word (" the" is a unit),
    so no learned token ever spans a word boundary. That means run.c's GLOBAL
    greedy merge and our PER-WORD greedy merge partition text identically —
    letting us cache per unique word and stay exactly equivalent.
"""
import argparse
import re
import struct
from collections import Counter
from functools import lru_cache

# Split into words (leading space attached), digit runs, punctuation runs,
# and whitespace runs. TinyStories is ~ASCII so this ASCII pattern suffices.
PAT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")

N_SPECIAL = 3          # <unk>, <s>, </s>
N_BASE = 256           # one token per byte
BASE = N_SPECIAL + N_BASE  # first merge id = 259


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(text, vocab_size, log_every=500):
    """Learn merges until we reach vocab_size. Returns (merges, ranks).
    merges: list of (id_a, id_b) in learn order; the new token id is BASE+i."""
    # Represent each unique word as a tuple of byte-token ids, with a frequency.
    words = Counter(PAT.findall(text))
    corpus = {tuple(b + N_SPECIAL for b in w.encode()): f for w, f in words.items()}
    print(f"bpe: {len(words):,} unique words, target vocab {vocab_size}")

    merges = []
    n_merges = vocab_size - BASE
    for i in range(n_merges):
        # count every adjacent pair, weighted by word frequency
        pairs = Counter()
        for word, freq in corpus.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        new_id = BASE + i
        merges.append((a, b))
        # apply the merge everywhere it occurs
        corpus = {_merge_word(word, a, b, new_id): freq for word, freq in corpus.items()}
        if (i + 1) % log_every == 0 or i == n_merges - 1:
            print(f"  merge {i+1:4d}/{n_merges}  vocab={new_id+1}")
    ranks = {pair: r for r, pair in enumerate(merges)}
    return merges, ranks


def _merge_word(word, a, b, new_id):
    if len(word) < 2:
        return word
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id); i += 2
        else:
            out.append(word[i]); i += 1
    return tuple(out)


# --------------------------------------------------------------------------
# vocab construction + encoding
# --------------------------------------------------------------------------
def build_vocab_bytes(merges):
    """id -> the raw bytes that token spells out."""
    vocab = {i: b"" for i in range(N_SPECIAL)}
    vocab[0], vocab[1], vocab[2] = b"<unk>", b"<s>", b"</s>"
    for b in range(256):
        vocab[N_SPECIAL + b] = bytes([b])
    for i, (a, b) in enumerate(merges):
        vocab[BASE + i] = vocab[a] + vocab[b]
    return vocab


class Tokenizer:
    def __init__(self, merges, ranks):
        self.merges = merges
        self.ranks = ranks
        self.vocab = build_vocab_bytes(merges)
        # (id_a, id_b) -> merged id, for the greedy encoder
        self.pair_to_id = {pair: BASE + i for i, pair in enumerate(merges)}
        self.encode_word = lru_cache(maxsize=None)(self._encode_word)

    def _encode_word(self, word: str):
        """Greedily merge the highest-priority (lowest-rank) pair until none
        apply — identical in effect to run.c's max-score merge loop."""
        ids = [b + N_SPECIAL for b in word.encode()]
        while len(ids) >= 2:
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            merged = self.pair_to_id[(ids[best_i], ids[best_i + 1])]
            ids[best_i:best_i + 2] = [merged]
        return ids

    def encode(self, text, bos=False, eos=False):
        """Mirror run.c: optional BOS, a dummy leading space, then per-word
        greedy merges."""
        out = [1] if bos else []
        if text:
            text = " " + text          # run.c's dummy-prefix space
        for word in PAT.findall(text):
            out.extend(self.encode_word(word))
        if eos:
            out.append(2)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---- run.c tokenizer.bin format ----
    def export(self, path):
        toks = [self.vocab[i] for i in range(len(self.vocab))]
        max_len = max(len(t) for t in toks)
        with open(path, "wb") as f:
            f.write(struct.pack("<i", max_len))
            for i, t in enumerate(toks):
                # score = -rank so the earliest-learned merge wins ties in
                # run.c's max-score loop; base/special tokens sit below all
                # merges so they're only leaves, never chosen as a merge.
                score = -(i - BASE) if i >= BASE else -1e6
                f.write(struct.pack("<fi", float(score), len(t)))
                f.write(t)
        print(f"wrote {path}: vocab={len(toks)}, max_token_len={max_len}")


def save_merges(merges, path):
    with open(path, "w") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")


def load_merges(path):
    merges = [tuple(map(int, line.split())) for line in open(path)]
    return merges, {pair: r for r, pair in enumerate(merges)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinystories_valid.txt")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--merges-out", default="models/tinystories.merges")
    ap.add_argument("--bin-out", default="models/tinystories_tokenizer.bin")
    ap.add_argument("--sample-mb", type=float, default=0.0,
                    help="cap training text (MB); 0 = use all")
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8", errors="replace").read()
    text = text.replace("<|endoftext|>", "\n")   # story delimiter -> whitespace
    if args.sample_mb:
        text = text[: int(args.sample_mb * 1_000_000)]
    print(f"training on {len(text)/1e6:.1f} MB")

    merges, ranks = train(text, args.vocab)
    save_merges(merges, args.merges_out)
    tok = Tokenizer(merges, ranks)
    tok.export(args.bin_out)

    # self-consistency: encode then decode must be identity (mod the dummy space)
    sample = "Once upon a time, there was a little robot who loved to read."
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"sample: {len(sample)} chars -> {len(ids)} tokens")
    print(f"  ids[:12]={ids[:12]}")
    print(f"  roundtrip={back!r}")
    assert back == " " + sample, "roundtrip mismatch!"
    print("  roundtrip OK")
