"""
PROMETHEUS — gen_prefs.py  (Phase 7, step 1 of 2)

Build a PREFERENCE dataset for DPO — the raw material of alignment.

Real RLHF collects pairs (prompt, chosen, rejected) where a *human* judged
chosen > rejected. We can't hand-label thousands of stories, so we use a
programmatic judge instead — which makes this RLAIF (AI/rule feedback) rather
than RLHF, but the DPO step that consumes these pairs is identical.

The judge is the exact weakness Phase 6 measured: does the story contain the
requested words? For each "use these words" instruction we sample K
completions FROM THE SFT MODEL ITSELF (on-policy), count how many requested
words each contains, and pair the best against the worst. When they differ,
that's a preference the model can learn from: "more words good, fewer bad."

    .venv/bin/python src/gen_prefs.py --prompts 1200 --k 8
    .venv/bin/python src/gen_prefs.py --prompts 60 --k 4 --out data/prefs_smoke.jsonl

Sampling K identical prompts as one batch keeps positions aligned (no padding,
no RoPE offset headaches) — the only variation across the K is the sampling
randomness, which is exactly the diversity we want.
"""
import argparse
import json
import random
import re
import time

import torch

from model import Transformer, ModelArgs
from bpe import Tokenizer, load_merges
from finetune import content_words, BOS


@torch.no_grad()
def sample_k(model, prompt_ids, k, device, max_new=180, temperature=0.9):
    """Draw k independent completions of one prompt. Returns list of k token
    lists (BOS-terminated turns, with the BOS stripped).

    Rows that hit BOS are DROPPED from the batch so we don't keep forwarding
    finished sequences — the active rows always share a length, so the tensor
    stays rectangular while it shrinks."""
    was_training = model.training
    model.eval()
    idx = torch.tensor([prompt_ids] * k, dtype=torch.int64, device=device)
    outs = [[] for _ in range(k)]
    active = list(range(k))            # original slot for each row of idx
    for _ in range(max_new):
        if not active:
            break
        logits, _ = model(idx[:, -model.args.max_seq_len:])
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)                 # (len(active), 1)
        idx = torch.cat([idx, nxt], dim=1)
        keep_rows, keep_slots = [], []
        for row, slot in enumerate(active):
            if int(nxt[row]) == BOS:
                continue                                  # finished — drop it
            outs[slot].append(int(nxt[row]))
            keep_rows.append(row)
            keep_slots.append(slot)
        if len(keep_rows) < len(active):
            idx = idx[keep_rows]                          # shrink to survivors
        active = keep_slots
    if was_training:
        model.train()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="models/tinystories_instruct.pt")
    ap.add_argument("--data", default="data/tinystories_train_300mb.txt")
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("--out", default="data/prefs.jsonl")
    ap.add_argument("--prompts", type=int, default=1200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tok = Tokenizer(*load_merges(args.merges))
    ck = torch.load(args.sft, map_location="cpu", weights_only=True)
    margs = ModelArgs(**ck["model_args"])
    model = Transformer(margs)
    model.load_state_dict(ck["model"])
    model.to(device)
    print(f"loaded SFT model: {sum(p.numel() for p in model.parameters()):,} params")

    # prompts drawn from the corpus, DISJOINT from the metric's held-out slice
    # (stories[:400], matching finetune.py) so before/after stays comparable.
    text = open(args.data, encoding="utf-8", errors="replace").read()
    stories = [s for s in re.split(r"<\|endoftext\|>", text) if s.strip()]
    random.shuffle(stories)
    pool = stories[400:]

    pairs, kept, seen, t0 = [], 0, 0, time.time()
    hist = {"chosen": 0, "rejected": 0}
    for story in pool:
        if kept >= args.prompts:
            break
        words = content_words(story.strip(), random.randint(2, 3))
        if len(words) < 2:
            continue
        seen += 1
        instr = f"Write a story using the words: {', '.join(words)}."
        prompt = f"User: {instr}\nAssistant:"
        prompt_ids = tok.encode(prompt, bos=True)

        completions = sample_k(model, prompt_ids, args.k, device,
                               temperature=args.temperature)
        scored = []
        for ids in completions:
            text_out = tok.decode(ids).strip()
            low = text_out.lower()
            score = sum(w in low for w in words)
            scored.append((score, text_out))
        scored.sort(key=lambda s: s[0])
        rej_score, rejected = scored[0]
        cho_score, chosen = scored[-1]

        # only a pair when the judge actually distinguishes them, and the
        # chosen text is a real story (avoid degenerate empty "wins")
        if cho_score > rej_score and len(chosen) > 40:
            pairs.append({"prompt": instr, "words": words,
                          "chosen": chosen, "rejected": rejected,
                          "chosen_score": cho_score, "rejected_score": rej_score})
            kept += 1
            hist["chosen"] += cho_score
            hist["rejected"] += rej_score
            if kept % 50 == 0:
                print(f"kept {kept:4d}/{args.prompts} pairs "
                      f"({seen} prompts seen) | {time.time()-t0:6.1f}s", flush=True)

    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    n = max(len(pairs), 1)
    print(f"\nwrote {len(pairs):,} preference pairs -> {args.out}")
    print(f"mean words present: chosen {hist['chosen']/n:.2f} vs "
          f"rejected {hist['rejected']/n:.2f} (of 2-3 requested)")


if __name__ == "__main__":
    main()
