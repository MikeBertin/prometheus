"""
PROMETHEUS — rm.py  (Phase 9, part 1 of 2)

A REWARD MODEL — the piece both the DPO and PPO toys deliberately skipped.

Real RLHF has three stages: SFT, then train a reward model on human preference
pairs, then RL against that reward model. DPO folds all three into one loss;
our Phase-8 PPO used the raw rule (word count) as the reward. Here we build the
real middle stage: a model that LEARNS to score responses from the preference
pairs, and outputs a continuous reward for any response.

Why bother, when the rule is right there? Two reasons that matter for Phase 9:
  - The rule is COARSE — with 2-3 words it can only return 0, ½, ⅓, ⅔, 1. A
    learned RM gives a smooth, dense score, which is a far better gradient
    signal for RL (part of why Phase-8 PPO couldn't climb).
  - It GENERALIZES — the RM scores partial progress and phrasing the rule can't
    see, exactly as a human-trained RM generalizes past its labels.

Architecture: the SFT transformer as a backbone + a scalar head on the LAST
token's hidden state (which has attended to the whole response). Trained with
the Bradley-Terry loss — the chosen response should score higher than the
rejected one:  loss = -log sigmoid(r(chosen) - r(rejected)).

    .venv/bin/python src/rm.py --epochs 3
"""
import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer, ModelArgs
from bpe import Tokenizer, load_merges
from finetune import BOS


class RewardModel(nn.Module):
    """SFT backbone + a scalar reward head. The reward is a MEAN-POOL of the
    hidden states over the RESPONSE tokens — a single summary token loses
    whether a specific word appeared 50 tokens back, but pooling aggregates that
    per-token signal, which is exactly what the word-inclusion preference needs."""
    def __init__(self, margs):
        super().__init__()
        self.backbone = Transformer(margs)
        self.head = nn.Linear(margs.dim, 1)
        nn.init.normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)

    def forward(self, tokens, mask):
        # tokens (N, T); mask (N, T) = 1 on real tokens (prompt+response), 0 pad.
        # Pool the WHOLE sequence: the head must see both which words were
        # requested (in the prompt) and whether they appear (in the response).
        _, _, hidden = self.backbone(tokens, return_hidden=True)      # (N, T, dim)
        m = mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)        # (N, dim)
        return self.head(pooled).squeeze(-1)                         # (N,)


def encode_side(tok, prompt, response, seq_len):
    ids = (tok.encode(f"User: {prompt}\nAssistant:", bos=True) +
           tok.encode(response, bos=False) + [BOS])[:seq_len]
    return ids


def collate(rows, device, pad=0):
    T = max(len(r) for r in rows)
    tok = torch.full((len(rows), T), pad, dtype=torch.int64, device=device)
    mask = torch.zeros(len(rows), T, dtype=torch.float32, device=device)
    for i, r in enumerate(rows):
        tok[i, :len(r)] = torch.tensor(r, device=device)
        mask[i, :len(r)] = 1.0                                        # all real tokens
    return tok, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="models/tinystories_instruct.pt")
    ap.add_argument("--prefs", default="data/prefs.jsonl")
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("--out", default="models/reward_model.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tok = Tokenizer(*load_merges(args.merges))
    ck = torch.load(args.sft, map_location="cpu", weights_only=True)
    margs = ModelArgs(**ck["model_args"])
    rm = RewardModel(margs)
    rm.backbone.load_state_dict(ck["model"])       # init backbone from SFT
    rm.to(device)
    print(f"reward model: {sum(p.numel() for p in rm.parameters()):,} params (SFT backbone + head)")

    pairs = [json.loads(l) for l in open(args.prefs)]
    data = [(encode_side(tok, p["prompt"], p["chosen"], margs.max_seq_len),
             encode_side(tok, p["prompt"], p["rejected"], margs.max_seq_len))
            for p in pairs]
    random.shuffle(data)
    n_val = int(len(data) * args.val_frac)
    val, train = data[:n_val], data[n_val:]
    print(f"pairs: {len(train)} train / {len(val)} val")

    opt = torch.optim.AdamW(rm.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def batch_scores(rows_pairs):
        ch, rj = zip(*rows_pairs)
        ct, cl = collate(list(ch), device)
        rt, rl = collate(list(rj), device)
        return rm(ct, cl), rm(rt, rl)

    @torch.no_grad()
    def accuracy(split):
        rm.eval(); correct = 0
        for s in range(0, len(split), args.batch):
            rc, rr = batch_scores(split[s:s + args.batch])
            correct += (rc > rr).sum().item()
        rm.train(); return correct / len(split)

    print(f"val accuracy before training: {accuracy(val):.1%}  (chance = 50%)")
    best_acc, step, t0 = 0.0, 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(train)
        for s in range(0, len(train), args.batch):
            rc, rr = batch_scores(train[s:s + args.batch])
            loss = -F.logsigmoid(rc - rr).mean()        # Bradley-Terry
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0); opt.step()
            step += 1
        acc = accuracy(val)
        flag = ""
        if acc > best_acc:                              # keep the val-best RM, not the last
            best_acc = acc
            torch.save({"backbone": rm.backbone.state_dict(),
                        "head": rm.head.state_dict(),
                        "model_args": margs.__dict__}, args.out)
            flag = " *"
        print(f"  epoch {epoch}: val accuracy {acc:.1%}{flag}", flush=True)
    print(f"saved {args.out} | best val accuracy {best_acc:.1%}")


if __name__ == "__main__":
    main()
