"""
PROMETHEUS — eval_lift.py

The honest version of the word-inclusion metric.

Every benchmark has a floor. TinyStories has a small, repetitive vocabulary,
so a story that ignored the instruction entirely still contains ~11% of the
requested words by coincidence. Reporting the raw score hides that; reporting
the LIFT over the floor is the number that means something.

The floor is measured with a shuffled control: score each generated story
against a DIFFERENT prompt's requested words. Same stories, same word
distribution, no instruction-following possible — whatever it scores is chance.

It also splits results by corpus frequency, because `content_words()` picks
*distinctive* words, so ~78% of what we ask for is rare — the headline is
dominated by the hardest version of the task.

    .venv/bin/python src/eval_lift.py                       # SFT vs DPO
    .venv/bin/python src/eval_lift.py --models models/tinystories_rloo.pt

Findings that produced walkthrough step 29:
    SFT   19.6% raw / 10.4% floor -> +9.2 lift  (common 42% | rare 13%)
    DPO   43.3% raw / 11.7% floor -> +31.7 lift (common 60% | rare 39%)
    => 9 -> 32 points of real signal, ~3.4x. The raw 22%->45% understated it.
    Standard error at n=240 is ~+/-3 points; small gaps are noise.
"""
import argparse
import math
import random
import re
from collections import Counter

import torch

from bpe import Tokenizer, load_merges
from finetune import content_words, generate_response
from dpo import load_model

COMMON_CUTOFF = 5000        # corpus occurrences to count a word as "common"
FREQ_SAMPLE = 20_000_000    # chars of corpus used for the frequency table


def build_requests(stories, n, seed=99):
    random.seed(seed)
    reqs = []
    for st in stories:
        w = content_words(st.strip(), 3)
        if len(w) >= 2:
            reqs.append(w)
        if len(reqs) >= n:
            break
    return reqs


def evaluate(path, label, reqs, tok, freq, device, temperature=0.6):
    model, _ = load_model(path, device, train=False)
    gens = [generate_response(model, tok,
                              f"Write a story using the words: {', '.join(w)}.",
                              device, temperature=temperature).lower()
            for w in reqs]

    hit = tot = 0
    buckets = {"common": [0, 0], "rare": [0, 0]}     # [hits, total]
    for w, g in zip(reqs, gens):
        for x in w:
            tot += 1
            ok = x in g
            hit += ok
            b = buckets["common" if freq[x] >= COMMON_CUTOFF else "rare"]
            b[1] += 1
            b[0] += ok

    # the floor: this story vs SOME OTHER prompt's words (offset avoids i==j)
    ch = ct = 0
    for i, g in enumerate(gens):
        for x in reqs[(i + 7) % len(reqs)]:
            ct += 1
            ch += (x in g)

    raw, floor = hit / tot, ch / ct
    se = math.sqrt(raw * (1 - raw) / tot)            # binomial standard error
    print(f"\n{label}")
    print(f"  raw            : {raw:.1%}  ({hit}/{tot})   +/- {se:.1%} (1 s.e.)")
    print(f"  floor (shuffled): {floor:.1%}  ({ch}/{ct})   <- instruction ignored")
    print(f"  LIFT           : {raw - floor:+.1%}   <- the real signal")
    for name, (h, t) in buckets.items():
        if t:
            print(f"  {name:>6} words   : {h/t:.1%}  (n={t})")
    return raw - floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["models/tinystories_instruct.pt",
                             "models/tinystories_aligned.pt"])
    ap.add_argument("--data", default="data/tinystories_train_300mb.txt")
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("-n", type=int, default=80, help="held-out prompts")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(*load_merges(args.merges))

    text = open(args.data, encoding="utf-8", errors="replace").read()
    stories = [s for s in re.split(r"<\|endoftext\|>", text) if s.strip()]
    random.Random(1234).shuffle(stories)             # same split as finetune.py
    held = stories[:400]                             # never trained on
    freq = Counter(re.findall(r"[a-z]{4,}", text[:FREQ_SAMPLE].lower()))

    reqs = build_requests(held, args.n)
    rare = sum(1 for w in reqs for x in w if freq[x] < COMMON_CUTOFF)
    total = sum(len(w) for w in reqs)
    print(f"{len(reqs)} prompts / {total} requested words "
          f"({rare/total:.0%} of them rare — this is a HARD benchmark)")

    for path in args.models:
        evaluate(path, path.split("/")[-1].replace(".pt", ""), reqs, tok, freq, device)


if __name__ == "__main__":
    main()
