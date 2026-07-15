"""
PROMETHEUS — ppo.py  (Phase 8: the RLHF path DPO shortcuts)

Proximal Policy Optimization — the classic RLHF algorithm (InstructGPT), here
as a toy so you can hold it next to dpo.py and SEE the difference.

Same goal as Phase 7 (get the model to use the requested words), same frozen
reference anchor. What changes is EVERYTHING about how we get there:

  DPO (dpo.py)                     PPO (this file)
  ------------------------------   ------------------------------------------
  offline preference PAIRS         on-policy ROLLOUTS, regenerated every step
  no reward model, no value net    a trainable VALUE HEAD (critic) + GAE
  one line of loss                 clipped surrogate + value loss + entropy + KL
  supervised, stable               reinforcement learning, finicky
  trains in ~90s                   generates fresh samples every iteration

Both use the SAME programmatic reward (fraction of requested words present).
Real RLHF replaces that rule with a REWARD MODEL trained on human preferences —
the one piece both our toys skip. What PPO adds over DPO is the RL itself:

  1. ROLLOUT — the current policy generates responses; we record each token's
     log-prob (the "old" policy) and the critic's value estimate.
  2. REWARD  — score each response (word inclusion), and shape a per-token
     reward = terminal reward at the end MINUS a KL penalty to the reference
     at every token (so the policy can't wander far from the SFT model).
  3. GAE     — the value head turns rewards into per-token ADVANTAGES.
  4. UPDATE  — for K epochs, the CLIPPED SURROGATE objective nudges the policy
     toward high-advantage actions, but clips the probability ratio so no
     single update moves too far (the "proximal" in PPO).

The critic is training-only; export saves just the policy, so run.c/runq.c run
it unchanged — zero C changes, same as every phase.

    .venv/bin/python src/ppo.py --iters 40
"""
import argparse
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer, ModelArgs
from bpe import Tokenizer, load_merges
from finetune import content_words, word_inclusion_metric, BOS


def load_policy(path, device, train):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    margs = ModelArgs(**ck["model_args"])
    m = Transformer(margs)
    m.load_state_dict(ck["model"])
    m.to(device)
    m.train(train)
    if not train:
        for p in m.parameters():
            p.requires_grad_(False)
    return m, margs


# ---------------------------------------------------------------------------
# 1. ROLLOUT — the policy generates, we record log-probs and values
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(policy, value_head, prompt_ids, words, tok, device, B, max_new):
    """Generate B responses to one prompt. Returns a list of B trajectories,
    each a dict of the tensors PPO needs. Batching B copies of the SAME prompt
    keeps positions aligned (no padding during generation).

    We sample from the policy's TRUE distribution (no temperature) so the
    recorded log-probs match what eval_actions recomputes — the PPO ratio must
    start at exactly 1 in the first epoch. Exploration comes from the softmax
    itself, not a temperature knob."""
    policy.eval()
    L = len(prompt_ids)
    seq = torch.tensor([prompt_ids] * B, dtype=torch.int64, device=device)
    actions = [[] for _ in range(B)]      # sampled token ids (incl. terminal BOS)
    old_logp = [[] for _ in range(B)]     # log-prob of each action under old policy
    values = [[] for _ in range(B)]       # V(state) before each action
    done = [False] * B
    for _ in range(max_new + 1):
        logits, _, hidden = policy(seq[:, -policy.args.max_seq_len:], return_hidden=True)
        logp = F.log_softmax(logits[:, -1, :], dim=-1)             # (B, V)
        v = value_head(hidden[:, -1, :]).squeeze(-1)               # (B,)
        nxt = torch.multinomial(logp.exp(), 1)                     # (B, 1)
        for i in range(B):
            if done[i]:
                continue
            t = int(nxt[i])
            actions[i].append(t)
            old_logp[i].append(float(logp[i, t]))
            values[i].append(float(v[i]))
            if t == BOS:
                done[i] = True            # emitting BOS is the terminal action
        seq = torch.cat([seq, nxt], dim=1)
        if all(done):
            break

    traj = []
    for i in range(B):
        text = tok.decode([t for t in actions[i] if t != BOS]).lower()
        reward = sum(w in text for w in words) / len(words)        # in [0, 1]
        terminated = actions[i] and actions[i][-1] == BOS
        traj.append({
            "full": prompt_ids + actions[i],   # prompt + response, for re-scoring
            "L": L,                            # first action is predicted at pos L-1
            "actions": actions[i],
            "old_logp": old_logp[i],
            "values": values[i],
            "reward": reward,
            "last_value": 0.0 if terminated else values[i][-1],  # bootstrap if cut
        })
    return traj


# ---------------------------------------------------------------------------
# 2. REWARD SHAPING + GAE
# ---------------------------------------------------------------------------
def shape_and_gae(traj, ref_logp, kl_coef, gamma, lam):
    """Per-token reward = -kl_coef * (old_logp - ref_logp), plus the terminal
    reward on the last action. Then GAE turns it into advantages/returns."""
    T = len(traj["actions"])
    rewards = [-kl_coef * (traj["old_logp"][t] - ref_logp[t]) for t in range(T)]
    rewards[-1] += traj["reward"]
    v = traj["values"]
    adv, gae = [0.0] * T, 0.0
    for t in reversed(range(T)):
        next_v = v[t + 1] if t + 1 < T else traj["last_value"]
        delta = rewards[t] + gamma * next_v - v[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = [adv[t] + v[t] for t in range(T)]
    return adv, returns


# ---------------------------------------------------------------------------
# 3. EVAL ACTIONS — recompute logp / value / entropy for a padded minibatch
# ---------------------------------------------------------------------------
def eval_actions(policy, value_head, batch, device):
    """For a list of trajectories, forward the full sequences once and gather,
    at each action's predicting position, the new log-prob, value and entropy.
    Returns (logp, value, entropy, mask) each (N, Amax)."""
    N = len(batch)
    full = [b["full"] for b in batch]
    Smax = max(len(f) for f in full)
    Amax = max(len(b["actions"]) for b in batch)
    seqs = torch.zeros(N, Smax, dtype=torch.int64, device=device)
    act_pos = torch.zeros(N, Amax, dtype=torch.int64, device=device)   # logits index
    act_tok = torch.zeros(N, Amax, dtype=torch.int64, device=device)
    mask = torch.zeros(N, Amax, device=device)
    for i, b in enumerate(batch):
        f, L, T = b["full"], b["L"], len(b["actions"])
        seqs[i, :len(f)] = torch.tensor(f, device=device)
        for j in range(T):
            act_pos[i, j] = L - 1 + j          # logits here predict action j
            act_tok[i, j] = b["actions"][j]
            mask[i, j] = 1.0

    logits, _, hidden = policy(seqs, return_hidden=True)            # (N,S,V),(N,S,d)
    gather_pos = act_pos.unsqueeze(-1).expand(-1, -1, logits.size(-1))
    a_logits = logits.gather(1, gather_pos)                         # (N,Amax,V)
    logp_all = F.log_softmax(a_logits.float(), dim=-1)
    logp = logp_all.gather(-1, act_tok.unsqueeze(-1)).squeeze(-1)   # (N,Amax)
    entropy = -(logp_all.exp() * logp_all).sum(-1)                  # (N,Amax)
    hpos = act_pos.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    value = value_head(hidden.gather(1, hpos)).squeeze(-1)          # (N,Amax)
    return logp, value, entropy, mask


# ---------------------------------------------------------------------------
# 4. PPO
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="models/tinystories_instruct.pt")
    ap.add_argument("--data", default="data/tinystories_train_300mb.txt")
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("--out", default="models/tinystories_ppo.bin")
    ap.add_argument("--ckpt", default="models/tinystories_ppo.pt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--prompts-per-iter", type=int, default=16)
    ap.add_argument("--samples", type=int, default=4, help="rollouts per prompt (B)")
    ap.add_argument("--epochs", type=int, default=4, help="PPO epochs per rollout")
    ap.add_argument("--minibatch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tok = Tokenizer(*load_merges(args.merges))
    policy, margs = load_policy(args.sft, device, train=True)
    ref, _ = load_policy(args.sft, device, train=False)
    value_head = nn.Linear(margs.dim, 1).to(device)
    nn.init.normal_(value_head.weight, std=0.01); nn.init.zeros_(value_head.bias)
    print(f"policy + frozen ref + value head ({sum(p.numel() for p in policy.parameters()):,} policy params)")

    opt = torch.optim.AdamW(list(policy.parameters()) + list(value_head.parameters()),
                            lr=args.lr, betas=(0.9, 0.95))

    # prompt pool (words are the reward target), disjoint from the eval held-out
    text = open(args.data, encoding="utf-8", errors="replace").read()
    stories = [s for s in re.split(r"<\|endoftext\|>", text) if s.strip()]
    random.Random(1234).shuffle(stories)
    held = stories[:400]
    pool = stories[400:]

    if not args.no_eval:
        before = word_inclusion_metric(ref, tok, held, device, n=args.eval_n)
        print(f"word-inclusion BEFORE (SFT): {before:.1%}")

    def make_prompt(story):
        w = content_words(story.strip(), random.randint(2, 3))
        if len(w) < 2:
            return None
        instr = f"Write a story using the words: {', '.join(w)}."
        return tok.encode(f"User: {instr}\nAssistant:", bos=True), w

    pi, t0 = 0, time.time()
    for it in range(args.iters):
        # ---- collect rollouts ----
        traj = []
        rewards_this_iter = []
        collected = 0
        while collected < args.prompts_per_iter:
            p = make_prompt(pool[pi % len(pool)]); pi += 1
            if p is None:
                continue
            prompt_ids, words = p
            samples = rollout(policy, value_head, prompt_ids, words, tok, device,
                              args.samples, args.max_new)
            collected += 1
            # ref log-probs for the KL term (one frozen forward per prompt-batch)
            with torch.no_grad():
                rl, _, _, _ = eval_actions(ref, value_head, samples, device)
            for k, tj in enumerate(samples):
                T = len(tj["actions"])
                adv, ret = shape_and_gae(tj, rl[k, :T].tolist(),
                                         args.kl_coef, args.gamma, args.lam)
                tj["adv"], tj["ret"] = adv, ret
                rewards_this_iter.append(tj["reward"])
            traj.extend(samples)

        # whiten advantages across the whole rollout (stabilizes PPO)
        all_adv = torch.tensor([a for tj in traj for a in tj["adv"]])
        amean, astd = all_adv.mean().item(), all_adv.std().item() + 1e-8
        for tj in traj:
            tj["adv"] = [(a - amean) / astd for a in tj["adv"]]

        # ---- PPO update: K epochs of clipped-surrogate over minibatches ----
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "n": 0}
        for _ in range(args.epochs):
            random.shuffle(traj)
            for s in range(0, len(traj), args.minibatch):
                mb = traj[s:s + args.minibatch]
                logp, value, entropy, mask = eval_actions(policy, value_head, mb, device)
                A = max(len(b["actions"]) for b in mb)
                old = torch.zeros(len(mb), A, device=device)
                adv = torch.zeros(len(mb), A, device=device)
                ret = torch.zeros(len(mb), A, device=device)
                for i, b in enumerate(mb):
                    T = len(b["actions"])
                    old[i, :T] = torch.tensor(b["old_logp"], device=device)
                    adv[i, :T] = torch.tensor(b["adv"], device=device)
                    ret[i, :T] = torch.tensor(b["ret"], device=device)

                ratio = torch.exp(logp - old)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                pg = -torch.min(unclipped, clipped)
                vf = 0.5 * (value - ret) ** 2
                m = mask.sum() + 1e-8
                pg_loss = (pg * mask).sum() / m
                vf_loss = (vf * mask).sum() / m
                ent = (entropy * mask).sum() / m
                loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_head.parameters()), 1.0)
                opt.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (logp - old))    # k3 KL estimator
                    stats["kl"] += (approx_kl * mask).sum().item() / m.item()
                stats["pg"] += pg_loss.item(); stats["vf"] += vf_loss.item()
                stats["ent"] += ent.item(); stats["n"] += 1

        n = stats["n"]
        print(f"iter {it:3d} | reward {sum(rewards_this_iter)/len(rewards_this_iter):.3f} "
              f"| pg {stats['pg']/n:+.4f} | vf {stats['vf']/n:.4f} "
              f"| ent {stats['ent']/n:.3f} | kl {stats['kl']/n:.4f} "
              f"| {time.time()-t0:6.1f}s", flush=True)

    if not args.no_eval:
        after = word_inclusion_metric(policy, tok, held, device, n=args.eval_n)
        print(f"\nword-inclusion  BEFORE (SFT): {before:.1%}")
        print(f"word-inclusion  AFTER  (PPO): {after:.1%}")

    print("\n=== PPO sample responses ===")
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
