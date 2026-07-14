"""
PROMETHEUS — finetune.py

Instruction fine-tuning (SFT) — turn the TinyStories *base* model into a
single-turn instruct model that follows a prompt instead of just continuing
text. Three real techniques, at 7M-param scale:

  1. A CHAT TEMPLATE (plain text — no new tokens, so run.c is unchanged):
         <s>User: <instruction>
         Assistant: <response></s>
     BOS (id 1) both opens the example and is the stop token run.c already
     halts on, so the model learns to end its turn.

  2. LOSS MASKING — we only train on the *response* tokens. The instruction
     is context, not something to learn to generate, so its target positions
     are set to -100 (PyTorch's cross_entropy ignore_index). Without this the
     model would waste capacity learning to parrot instructions.

  3. SFT FROM THE PRETRAINED BASE — we start from models/tinystories.pt, not
     random weights. The base already knows how to write stories; fine-tuning
     only teaches it the *format* of following an instruction. Hence a low LR
     and a couple of epochs, not a full training run.

The instruction data is SYNTHESIZED from the TinyStories corpus itself:
extract content words from a story, and the instruction becomes "write a
story using these words" — with the story as the target response. That makes
the task in-domain AND verifiable (does the output contain the words?).

We full-fine-tune all 7M parameters — trivial at this scale. Real instruct
models use LoRA/PEFT to update a small adapter instead; noted, not needed here.

  .venv/bin/python src/finetune.py                 # full run
  .venv/bin/python src/finetune.py --max-examples 200 --epochs 1   # smoke test
"""
import argparse
import math
import random
import re
import time

import numpy as np
import torch

from model import Transformer, ModelArgs
from bpe import Tokenizer, load_merges
from export import legacy_export

BOS = 1
STOPWORDS = set("""
the a an and or but so then they them their there here that this these those with
without from into onto over under they was were been being have has had will would
could should shall can may might must not you your yours she her his him he it its
one two out for are our who what when where why how all any some more most very just
day time went said saw came got had did too also him her them then once upon big
little went want like love play day them each said name named around after before
""".split())

PROMPT_TEMPLATES = [
    ("words", "Write a story using the words: {words}."),
    ("words", "Write a short story that includes these words: {words}."),
    ("about", "Write a story about a {topic}."),
    ("plain", "Write a short story."),
    ("plain", "Tell me a story."),
]


def content_words(story, k):
    """Pick up to k distinctive content words that appear in the story."""
    words = re.findall(r"[a-z]{4,}", story.lower())
    seen, pool = set(), []
    for w in words:
        if w not in STOPWORDS and w not in seen:
            seen.add(w); pool.append(w)
    random.shuffle(pool)
    return pool[:k]


def make_instruction(story):
    """Return an instruction string, or None to skip this story."""
    kind, tmpl = random.choice(PROMPT_TEMPLATES)
    if kind == "words":
        w = content_words(story, random.randint(2, 3))
        if len(w) < 2:
            return "Write a short story."
        return tmpl.format(words=", ".join(w))
    if kind == "about":
        w = content_words(story, 1)
        if not w:
            return "Write a short story."
        return tmpl.format(topic=w[0])
    return tmpl


def build_examples(stories, tok, seq_len, max_examples):
    """Tokenize each (instruction, story) pair into padded (x, y) rows with the
    instruction masked out. y uses -100 everywhere loss should be ignored."""
    X = np.zeros((max_examples, seq_len), dtype=np.int32)
    Y = np.full((max_examples, seq_len), -100, dtype=np.int32)
    n = 0
    for story in stories:
        story = story.strip()
        if not story:
            continue
        instr = make_instruction(story)
        prompt = f"User: {instr}\nAssistant:"
        prompt_ids = tok.encode(prompt, bos=True)          # [BOS, ...:]
        resp_ids = tok.encode(story, bos=False)            # [ , Once, ...]
        full = prompt_ids + resp_ids + [BOS]               # BOS = stop token
        if len(full) > seq_len + 1:
            continue                                       # doesn't fit context
        x = full[:-1]
        y = full[1:]
        # mask: only positions predicting a response token count toward loss.
        # y[i] is a response token once i >= len(prompt_ids) - 1.
        mask_upto = len(prompt_ids) - 1
        X[n, :len(x)] = x
        for i in range(mask_upto, len(y)):
            Y[n, i] = y[i]
        n += 1
        if n >= max_examples:
            break
    print(f"built {n:,} SFT examples (seq_len {seq_len})")
    return X[:n], Y[:n]


@torch.no_grad()
def generate_response(model, tok, instruction, device, max_new=200, temperature=0.7):
    """Prompt the model with the chat template and stop at BOS."""
    was_training = model.training   # restore, don't force-.train() — a frozen
    model.eval()                    # reference passed in must stay frozen
    prompt = f"User: {instruction}\nAssistant:"
    ids = tok.encode(prompt, bos=True)
    idx = torch.tensor([ids], dtype=torch.int64, device=device)
    out = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.args.max_seq_len:])
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        t = nxt.item()
        if t == BOS:
            break
        out.append(t)
        idx = torch.cat([idx, nxt], dim=1)
    if was_training:
        model.train()
    return tok.decode(out).strip()


def word_inclusion_metric(model, tok, stories, device, n=100):
    """For 'use these words' instructions, what fraction of requested words
    actually appear in the generated story? The honest quality number."""
    hit, tot = 0, 0
    for story in stories[:n]:
        w = content_words(story.strip(), 3)
        if len(w) < 2:
            continue
        instr = f"Write a story using the words: {', '.join(w)}."
        gen = generate_response(model, tok, instr, device, temperature=0.6).lower()
        for word in w:
            tot += 1
            hit += (word in gen)
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/tinystories.pt")
    ap.add_argument("--data", default="data/tinystories_train_300mb.txt")
    ap.add_argument("--merges", default="models/tinystories.merges")
    ap.add_argument("--out", default="models/tinystories_instruct.bin")
    ap.add_argument("--ckpt", default="models/tinystories_instruct.pt")
    ap.add_argument("--max-examples", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tok = Tokenizer(*load_merges(args.merges))

    # load the pretrained base (architecture comes from its own saved args)
    ck = torch.load(args.base, map_location="cpu", weights_only=True)
    margs = ModelArgs(**ck["model_args"])
    model = Transformer(margs)
    model.load_state_dict(ck["model"])
    model.to(device)
    print(f"loaded base: {sum(p.numel() for p in model.parameters()):,} params")

    # synthesize the SFT set from the corpus
    text = open(args.data, encoding="utf-8", errors="replace").read()
    stories = [s for s in re.split(r"<\|endoftext\|>", text) if s.strip()]
    random.shuffle(stories)
    held_out = stories[:400]                 # for the metric, never trained on
    X, Y = build_examples(stories[400:], tok, margs.max_seq_len, args.max_examples)
    X = torch.from_numpy(X.astype(np.int64))
    Y = torch.from_numpy(Y.astype(np.int64))

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    n = X.shape[0]
    iters_per_epoch = n // args.batch
    total_iters = iters_per_epoch * args.epochs
    print(f"fine-tuning: {n:,} examples, {args.epochs} epochs, {total_iters:,} iters")

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        for b in range(iters_per_epoch):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            x, y = X[idx].to(device), Y[idx].to(device)
            _, loss = model(x, y)            # cross_entropy ignores -100 targets
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # cosine LR with short warmup
            lr = args.lr * (min(1.0, (step + 1) / 50)) * \
                (0.5 * (1 + math.cos(math.pi * step / total_iters)))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            step += 1
            if step % 100 == 0 or step == total_iters:
                print(f"epoch {epoch} step {step:5d}/{total_iters} | "
                      f"loss {loss.item():.4f} | {time.time()-t0:6.1f}s", flush=True)

    # eyeball a few responses
    print("\n=== sample responses ===")
    for instr in ["Write a story using the words: dragon, cake, brave.",
                  "Write a story about a robot.",
                  "Tell me a story."]:
        print(f"\n> {instr}\n{generate_response(model, tok, instr, device)}")

    rate = word_inclusion_metric(model, tok, held_out, device, n=120)
    print(f"\nword-inclusion rate on held-out instructions: {rate:.1%}")

    torch.save({"model": model.state_dict(), "model_args": margs.__dict__}, args.ckpt)
    model.cpu()
    legacy_export(model, args.out)
    print(f"saved {args.ckpt} + {args.out}")


if __name__ == "__main__":
    main()
