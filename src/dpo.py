"""
PROMETHEUS — dpo.py  (Phase 7, step 2 of 2)

Direct Preference Optimization — the alignment stage, after SFT.

Classic RLHF trains a separate reward model on the preference pairs, then uses
RL (PPO) to push the policy toward high reward. DPO (Rafailov et al., 2023)
proves you can skip both: a preference pair is optimized directly with a
simple classification-style loss, and the "reward" is implicit in the policy's
own log-probs relative to a frozen reference.

Two copies of the SFT model:
  - policy    (trainable)  — what we're aligning
  - reference (frozen)     — the SFT model, an anchor so the policy can't drift
                             into gibberish just to win the preference

For each pair (prompt, chosen, rejected), with sequence log-probs summed over
ONLY the response tokens (same masking idea as SFT):

  logits = beta * [ (logp_pol(chosen) - logp_pol(rejected))       <- policy prefers?
                  - (logp_ref(chosen) - logp_ref(rejected)) ]     <- beyond the reference
  loss   = -log sigmoid(logits)

Raising chosen's probability and lowering rejected's minimizes the loss; the
reference term means we only get rewarded for preferences the SFT model didn't
already have. beta controls how far the policy may stray from the reference.

    .venv/bin/python src/dpo.py --prefs data/prefs.jsonl --epochs 2
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from model import Transformer, ModelArgs
from bpe import Tokenizer, load_merges
from finetune import BOS, word_inclusion_metric
import re


def load_model(path, device, train):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    margs = ModelArgs(**ck["model_args"])
    m = Transformer(margs)
    m.load_state_dict(ck["model"])
    m.to(device)
    if train:
        m.train()
    else:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m, margs


def encode_pair(tok, ex, seq_len):
    """(tokens, response-mask) for chosen and rejected. Mask is over TARGET
    positions (len T-1): True where the predicted token is part of the response
    (incl. the final BOS stop), False over the prompt and any padding."""
    prompt = f"User: {ex['prompt']}\nAssistant:"
    p_ids = tok.encode(prompt, bos=True)
    out = {}
    for key in ("chosen", "rejected"):
        r_ids = tok.encode(ex[key], bos=False) + [BOS]
        full = (p_ids + r_ids)[:seq_len]
        mask = [i >= len(p_ids) - 1 for i in range(len(full) - 1)]  # target-aligned
        out[key] = (full, mask)
    return out


def collate(batch, key, device, pad=0):
    """Right-pad a batch's chosen/ rejected side into tensors."""
    seqs = [b[key][0] for b in batch]
    masks = [b[key][1] for b in batch]
    T = max(len(s) for s in seqs)
    tok = torch.full((len(seqs), T), pad, dtype=torch.int64)
    msk = torch.zeros((len(seqs), T - 1), dtype=torch.bool)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        tok[i, :len(s)] = torch.tensor(s)
        msk[i, :len(m)] = torch.tensor(m)
    return tok.to(device), msk.to(device)


def seq_logprob(model, tokens, mask):
    """Sum of log p(token | prefix) over the masked (response) positions."""
    logits, _ = model(tokens)                              # (B, T, V)
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tok_logp = logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)  # (B, T-1)
    return (tok_logp * mask).sum(-1)                       # (B,)


def dpo_loss(policy, ref, batch, device, beta):
    ct, cm = collate(batch, "chosen", device)
    rt, rm = collate(batch, "rejected", device)
    pol_ch, pol_rj = seq_logprob(policy, ct, cm), seq_logprob(policy, rt, rm)
    with torch.no_grad():
        ref_ch, ref_rj = seq_logprob(ref, ct, cm), seq_logprob(ref, rt, rm)
    logits = beta * ((pol_ch - pol_rj) - (ref_ch - ref_rj))
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean()          # policy ranks chosen over rejected
    margin = (pol_ch - pol_rj).mean()          # implicit reward margin
    return loss, acc.item(), margin.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="models/tinystories_instruct.pt")
    ap.add_argument("--prefs", default="data/prefs.jsonl")
    ap.add_argument("--data", default="data/tinystories_train_300mb.txt")
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("--out", default="models/tinystories_aligned.bin")
    ap.add_argument("--ckpt", default="models/tinystories_aligned.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tok = Tokenizer(*load_merges(args.merges))
    policy, margs = load_model(args.sft, device, train=True)
    ref, _ = load_model(args.sft, device, train=False)
    print(f"policy + frozen reference loaded ({sum(p.numel() for p in policy.parameters()):,} params each)")

    pairs = [json.loads(l) for l in open(args.prefs)]
    data = [encode_pair(tok, ex, margs.max_seq_len) for ex in pairs]
    print(f"preference pairs: {len(data):,}")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    # held-out slice for the honest before/after metric (never in the pref set)
    held = None
    if not args.no_eval:
        text = open(args.data, encoding="utf-8", errors="replace").read()
        stories = [s for s in re.split(r"<\|endoftext\|>", text) if s.strip()]
        random.Random(1234).shuffle(stories)     # same seed/order as finetune.py
        held = stories[:400]
        before = word_inclusion_metric(ref, tok, held, device, n=args.eval_n)
        print(f"word-inclusion BEFORE (SFT): {before:.1%}")

    n = len(data)
    total = (n // args.batch) * args.epochs
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(data)
        for b in range(n // args.batch):
            batch = data[b * args.batch:(b + 1) * args.batch]
            loss, acc, margin = dpo_loss(policy, ref, batch, device, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == total:
                print(f"epoch {epoch} step {step:4d}/{total} | loss {loss.item():.4f} "
                      f"| acc {acc:.2f} | margin {margin:+.2f} | {time.time()-t0:6.1f}s",
                      flush=True)

    if not args.no_eval:
        after = word_inclusion_metric(policy, tok, held, device, n=args.eval_n)
        print(f"\nword-inclusion  BEFORE (SFT): {before:.1%}")
        print(f"word-inclusion  AFTER  (DPO): {after:.1%}")

    print("\n=== aligned sample responses ===")
    from finetune import generate_response
    for instr in ["Write a story using the words: dragon, cake, brave.",
                  "Write a story using the words: moon, boat, secret.",
                  "Write a story about a robot."]:
        print(f"\n> {instr}\n{generate_response(policy, tok, instr, device)}")

    torch.save({"model": policy.state_dict(), "model_args": margs.__dict__}, args.ckpt)
    policy.cpu()
    from export import legacy_export
    legacy_export(policy, args.out)
    print(f"\nsaved {args.ckpt} + {args.out}")


if __name__ == "__main__":
    main()
