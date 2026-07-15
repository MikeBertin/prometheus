"""
PROMETHEUS — rloo.py  (Phase 9, part 2 of 2)

RLOO — REINFORCE Leave-One-Out. The modern, value-net-free way to do RLHF, and
Phase 9's answer to Phase 8: naive PPO couldn't climb; RLOO + a learned reward
model does. Two changes fix what was broken:

  1. A learned REWARD MODEL (rm.py) instead of the coarse rule — a dense,
     smooth reward, which is a far better gradient signal.
  2. A LEAVE-ONE-OUT baseline instead of PPO's value head + GAE. For each prompt
     we draw k samples; the baseline for sample i is the mean reward of the
     OTHER k-1 samples. This is an unbiased, low-variance advantage estimate
     with no critic to train, no GAE, no clipping — the machinery PPO needed and
     RLOO throws away. (Cohere's RLOO; the same family as GRPO.)

Per prompt, for each of the k samples:
    reward_i    = RM(response_i) - kl_coef * KL(policy || ref)_i   # dense reward, tethered
    baseline_i  = mean(reward_j : j != i)                          # leave-one-out
    advantage_i = reward_i - baseline_i
    loss       += -advantage_i * sum_t log p_policy(token_t)       # plain REINFORCE

No old-policy ratio, no clip: RLOO does one on-policy update per rollout. The
advantage is detached (it's the reward signal); the gradient flows only through
the freshly recomputed log-probs. The reference model still anchors via the KL
term, and the critic is gone entirely.

    .venv/bin/python src/rloo.py --iters 40
"""
import argparse
import random
import re
import time

import torch
import torch.nn.functional as F

from model import Transformer, ModelArgs
from bpe import Tokenizer, load_merges
from finetune import content_words, word_inclusion_metric, generate_response, BOS
from rm import RewardModel


def load_policy(path, device, train):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    margs = ModelArgs(**ck["model_args"])
    m = Transformer(margs); m.load_state_dict(ck["model"]); m.to(device)
    m.train(train)
    if not train:
        for p in m.parameters():
            p.requires_grad_(False)
    return m, margs


def load_rm(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    rm = RewardModel(ModelArgs(**ck["model_args"]))
    rm.backbone.load_state_dict(ck["backbone"]); rm.head.load_state_dict(ck["head"])
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm


@torch.no_grad()
def rollout(policy, prompt_ids, k, device, max_new):
    """k on-policy samples of one prompt. True-distribution sampling (no temp)."""
    policy.eval()
    L = len(prompt_ids)
    seq = torch.tensor([prompt_ids] * k, dtype=torch.int64, device=device)
    actions = [[] for _ in range(k)]
    done = [False] * k
    for _ in range(max_new + 1):
        logits, _ = policy(seq[:, -policy.args.max_seq_len:])
        nxt = torch.multinomial(F.log_softmax(logits[:, -1, :], -1).exp(), 1)
        for i in range(k):
            if done[i]:
                continue
            t = int(nxt[i]); actions[i].append(t)
            if t == BOS:
                done[i] = True
        seq = torch.cat([seq, nxt], dim=1)
        if all(done):
            break
    return [{"full": prompt_ids + a, "L": L, "actions": a} for a in actions]


def sum_response_logp(model, batch, device):
    """Sum of log p(action) over each trajectory's response tokens. Grad flows
    iff model is trainable. Returns (N,)."""
    N = len(batch)
    Smax = max(len(b["full"]) for b in batch)
    seqs = torch.zeros(N, Smax, dtype=torch.int64, device=device)
    for i, b in enumerate(batch):
        seqs[i, :len(b["full"])] = torch.tensor(b["full"], device=device)
    logits, _ = model(seqs)
    logp_all = F.log_softmax(logits.float(), dim=-1)
    out = torch.zeros(N, device=device)
    for i, b in enumerate(batch):
        L, T = b["L"], len(b["actions"])
        pos = torch.arange(L - 1, L - 1 + T, device=device)          # predicting positions
        act = torch.tensor(b["actions"], device=device)
        out[i] = logp_all[i, pos].gather(-1, act.unsqueeze(-1)).squeeze(-1).sum()
    return out


@torch.no_grad()
def rm_scores(rm, batch, device):
    N = len(batch)
    Smax = max(len(b["full"]) for b in batch)
    seqs = torch.zeros(N, Smax, dtype=torch.int64, device=device)
    mask = torch.zeros(N, Smax, dtype=torch.float32, device=device)
    for i, b in enumerate(batch):
        seqs[i, :len(b["full"])] = torch.tensor(b["full"], device=device)
        mask[i, :len(b["full"])] = 1.0                               # whole sequence
    return rm(seqs, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="models/tinystories_instruct.pt")
    ap.add_argument("--rm", default="models/reward_model.pt")
    ap.add_argument("--data", default="data/tinystories_train_300mb.txt")
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("--out", default="models/tinystories_rloo.bin")
    ap.add_argument("--ckpt", default="models/tinystories_rloo.pt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--prompts-per-iter", type=int, default=16)
    ap.add_argument("--k", type=int, default=6, help="samples per prompt (leave-one-out)")
    ap.add_argument("--minibatch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tok = Tokenizer(*load_merges(args.merges))
    policy, margs = load_policy(args.sft, device, train=True)
    ref, _ = load_policy(args.sft, device, train=False)
    rm = load_rm(args.rm, device)
    print(f"policy (trainable) + frozen ref + frozen reward model loaded")
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))

    text = open(args.data, encoding="utf-8", errors="replace").read()
    stories = [s for s in re.split(r"<\|endoftext\|>", text) if s.strip()]
    random.Random(1234).shuffle(stories)
    held, pool = stories[:400], stories[400:]

    if not args.no_eval:
        before = word_inclusion_metric(ref, tok, held, device, n=args.eval_n)
        print(f"word-inclusion BEFORE (SFT): {before:.1%}")

    def make_prompt(story):
        w = content_words(story.strip(), random.randint(2, 3))
        if len(w) < 2:
            return None
        return tok.encode(f"User: Write a story using the words: {', '.join(w)}.\nAssistant:", bos=True)

    pi, t0 = 0, time.time()
    for it in range(args.iters):
        # ---- collect k samples for each of prompts_per_iter prompts ----
        groups = []                      # list of k-sample trajectory lists
        while len(groups) < args.prompts_per_iter:
            prompt_ids = None
            while prompt_ids is None:
                prompt_ids = make_prompt(pool[pi % len(pool)]); pi += 1
            groups.append(rollout(policy, prompt_ids, args.k, device, args.max_new))

        flat = [t for g in groups for t in g]
        with torch.no_grad():
            rmv = rm_scores(rm, flat, device)                        # (N,)
            oldlp = sum_response_logp(policy, flat, device)          # (N,) old policy
            reflp = sum_response_logp(ref, flat, device)             # (N,) reference
        kl = oldlp - reflp                                           # seq-level KL est
        reward = rmv - args.kl_coef * kl

        # leave-one-out advantage within each k-group, then whiten
        adv = torch.zeros_like(reward)
        mean_rm = rmv.mean().item()
        for gi, g in enumerate(groups):
            idx = list(range(gi * args.k, gi * args.k + len(g)))
            r = reward[idx]
            loo = (r.sum() - r) / (len(g) - 1)
            adv[idx] = r - loo
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- one REINFORCE update (minibatched) over the rollout ----
        pg = klm = 0.0; nb = 0
        order = list(range(len(flat))); random.shuffle(order)
        for s in range(0, len(order), args.minibatch):
            mb_idx = order[s:s + args.minibatch]
            mb = [flat[i] for i in mb_idx]
            logp = sum_response_logp(policy, mb, device)             # grad flows
            a = adv[mb_idx]
            loss = -(a.detach() * logp / 100.0).mean()               # /100 ~ per-token scale
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            pg += loss.item(); nb += 1
        klm = kl.mean().item()
        print(f"iter {it:3d} | RM {mean_rm:+.3f} | reward {reward.mean().item():+.3f} "
              f"| kl {klm:+.3f} | loss {pg/nb:+.4f} | {time.time()-t0:6.1f}s", flush=True)

    if not args.no_eval:
        after = word_inclusion_metric(policy, tok, held, device, n=args.eval_n)
        print(f"\nword-inclusion  BEFORE (SFT):  {before:.1%}")
        print(f"word-inclusion  AFTER  (RLOO): {after:.1%}")

    print("\n=== RLOO sample responses ===")
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
