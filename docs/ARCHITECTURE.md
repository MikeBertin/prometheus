# PROMETHEUS — Architecture Notes

A map from the transformer math to the lines in [`src/run.c`](../src/run.c).
This is a **decoder-only transformer**, Llama-2 flavor, inference only.

## The big picture

A language model is a function: given a sequence of tokens, output a
probability distribution over what the next token should be. Generation is
just calling that function in a loop, sampling one token, appending it, and
calling again.

```
token id ──► [embedding] ──► residual stream x (a vector of size `dim`)
                                   │
              ┌────────────────────┴───────────────────┐
              │  repeat n_layers times:                 │
              │    x += Attention(RMSNorm(x))           │  ← mixes across tokens
              │    x += FeedForward(RMSNorm(x))         │  ← thinks per token
              └────────────────────┬───────────────────┘
                                   │
        [RMSNorm] ──► [linear classifier wcls] ──► logits (size vocab_size)
                                   │
                          [sample] ──► next token id
```

The **residual stream** `x` is the throughline: every sub-layer reads it,
computes a correction, and adds that correction back. Information is never
overwritten, only edited — that's why deep transformers train stably.

## The pieces, and where they live

| Concept | What it does | Code |
|---|---|---|
| **Embedding** | token id → learned vector | `forward()`, `memcpy` from `token_embedding_table` |
| **RMSNorm** | rescale a vector to unit RMS, then per-dim gain. Stabilizes scale before each sub-layer. No mean-subtraction, no bias (cheaper than LayerNorm). | `rmsnorm()` |
| **QKV projection** | three matmuls turn `x` into query, key, value vectors | `matmul` of `wq/wk/wv` |
| **RoPE** | rotary position encoding — *rotates* each (even,odd) dim pair by an angle ∝ position. Attention dot-products then depend only on *relative* distance between tokens. | the `for (i ... i+=2)` loop |
| **Attention** | each head: score every past token by query·key, softmax, take weighted sum of values. This is the only place tokens talk to each other. | the `for (h ...)` head loop |
| **KV cache** | past keys/values are stored, so step *t* is O(t) not O(t²). | `key_cache` / `value_cache` |
| **GQA** | query heads can share key/value heads (`kv_mul`). Saves cache memory. (stories15M uses 1:1, so kv_mul=1.) | `h / kv_mul` indexing |
| **SwiGLU FFN** | `w2( silu(w1·x) * (w3·x) )`. The gated activation is where most parameters and most "knowledge" live. | the `hidden_dim` loop |
| **Classifier** | final matmul to vocab-sized logits. Weights often *tied* to the embedding table. | `matmul(logits, x, wcls...)` |
| **Sampling** | temperature scales logits; top-p keeps the smallest set of tokens summing to p, then draws one. | `sample()` / `sample_topp()` |

## Why attention is the heart of it

Everything except the `for (h ...)` head loop operates on each token
independently. Attention is the *only* operation that moves information
*between* token positions. The query asks "what am I looking for?", each past
key answers "here's what I am", and the softmax-weighted sum of values pulls in
the relevant context. "Attention Is All You Need" is literally true: strip
attention and you have a fancy per-token MLP with no memory.

## The causal mask, for free

We never build a mask matrix. The KV cache only ever contains positions
`0..pos`, and the score loop runs `for (t = 0; t <= pos; t++)`. A token
physically cannot attend to the future because the future isn't in the cache
yet. The autoregressive constraint is structural, not enforced.

## stories15M dimensions (for grounding)

```
dim=288  hidden_dim=768  n_layers=6  n_heads=6  n_kv_heads=6
head_size=48  vocab_size=32000  seq_len=256   (~15M parameters)
```

## What's deliberately NOT here

- **No backward pass / autograd.** This is inference. Gradients, the loss, and
  the optimizer live in the (future) `train.py`.
- **No batching.** One sequence at a time — clearest for learning.
- **No quantization.** Pure float32. A `runq.c` int8 variant is a good later exercise.

## Phase 2 — training our own weights

The pipeline: [`train.py`](../src/train.py) trains
[`model.py`](../src/model.py) (the PyTorch twin of `run.c`) →
[`export.py`](../src/export.py) writes the binary layout `read_checkpoint()`
expects → the same `run.c` runs *our* weights.

**The format contract** is the `memory_map_weights()` ordering — header of 7
ints, then each tensor type concatenated across layers, in order. `nn.Linear`
stores weights as (out, in) row-major, which is exactly the layout `matmul()`
indexes, so tensors dump with no transpose.

**What must match between model.py and run.c** (get any of these wrong and
the export produces fluent garbage):

| Knob | Value | Where |
|---|---|---|
| RMSNorm eps | `1e-5` | `rmsnorm()` ↔ `RMSNorm` |
| RoPE base | `10000` | the freq formula in both |
| RoPE pairing | **adjacent** dims (i, i+1) | run.c's `i += 2` loop ↔ `view_as_complex` on reshape(..., -1, 2). *Not* the HF "rotate-half" convention. |
| Attention scale | `1/sqrt(head_size)` | explicit in run.c, SDPA default in torch |
| Tied classifier | yes → **positive** vocab_size in header | `shared_weights` flag |

**The byte-level tokenizer** ([`tokenizer_export.py`](../src/tokenizer_export.py)):
vocab = `<unk>`, `<s>`, `</s>` + one token per byte (259 total). run.c needed
zero changes — printable bytes are their own token strings, the rest use the
`<0xNN>` spelling `decode()` already parses, and the BPE merge loop simply
finds nothing to merge. Trade: no tokenizer training at all, but 1 byte = 1
token, so `seq_len=256` sees ~256 characters of context.

**Phase-2 model** (`ModelArgs` defaults): dim=192, 5 layers, 6 heads,
hidden=512, seq_len=256, vocab=259 → **2.26M params**, trained on
tiny-Shakespeare (~1.1M tokens) with AdamW + cosine LR on Apple MPS.
BOS is inserted at every blank-line boundary so the model learns it as
"new speaker block" — which is why unprompted generation (run.c starts from
BOS) opens like a play instead of mid-sentence.

## Phase 3 — the web demo (`make web`)

The same `run.c`, compiled to WebAssembly, generating live in the browser at
[web/index.html](../web/index.html) — hermes/kernel-style page: live demo in
the hero + 15-step annotated walkthrough (including the real loss curves).

- `run.c`'s CLI `main()` is guarded by `#ifndef PROMETHEUS_LIB`;
  [`src/web_api.c`](../src/web_api.c) includes it and exposes a 4-function
  stateful API (`prom_init` / `prom_start` / `prom_next` / `prom_is_done`).
  JS drives generation one token at a time.
- JS fetches the `.bin`s (progress bar), writes them into emscripten's MEMFS,
  and `mmap()`/`open()` in run.c work unchanged.
- Build: `emcc -O3 -msimd128 -ffast-math … -sMODULARIZE -sEXPORT_NAME=Prometheus
  -sEXPORTED_RUNTIME_METHODS=cwrap,FS -sALLOW_MEMORY_GROWTH --no-entry`.
- The page benchmarks the raw engine at load (a silent flat-out generation)
  and *paces* the visible typewriter at ~100 tok/s wall-clock — pacing by
  elapsed time means throttled rAF frames burst-catch-up instead of crawling.
- Serve locally: `python3 -m http.server --directory web` (or the
  `prometheus-web` entry in `.claude/launch.json`).

## Phase 4 — int8 quantization (`runq.c`)

[`src/runq.c`](../src/runq.c) is run.c's quantized sibling: same transformer,
but every matmul weight stored as **Q8_0** — int8 values + one fp32 scale per
group of GS consecutive values (`scale = max|w|/127`, symmetric). Activations
are quantized on the fly before each matmul; the inner dot product accumulates
in int32 and applies the two scales once per group. Norms, activations
between ops, the KV cache and logits stay fp32.

Why it wins: matmul is memory-bandwidth-bound, and int8 moves 4× fewer bytes.
Measured (M-series, `-n 256`): Shakespeare 9.1→2.4 MB and ~6.7k→~15.8k tok/s
(2.3×); stories15M 60.8→17.1 MB and 944→1,572 tok/s (1.7×).

Export: `export.py --q80` writes the v2 "ak42" format (256-byte header with
`shared_classifier` + `group_size`, fp32 norms first, then each tensor as
`[int8 q][fp32 scales]`). It accepts either `ckpt.pt` **or a legacy fp32
.bin** — so stories15M can be quantized without its original checkpoint.

The web demo now ships this engine: `make web` compiles `web_api.c` with
`-DPROMETHEUS_Q` (which includes runq.c instead of run.c — identical
internals) and the page fetches `shakespeare_q80.bin` (2.4 MB vs 9.1).
Walkthrough steps 16–17 cover Q8_0 and the row-alignment war story.

**The gotcha that cost an hour**: groups must align with matrix *rows*.
runq.c's matmul indexes a weight's scales as `(row*n + j)/GS`, valid only if
GS divides every row length (dim *and* hidden_dim). stories15M has dim=288 —
not divisible by 64 — so at GS=64 every odd row read its neighbour's scales
and the model emitted intermittent junk tokens ("named**opts**a") while
mostly-fluent text survived. Shakespeare (dim=192) masked the bug entirely.
The exporter now auto-shrinks GS (64→32 for stories15M) and runq.c refuses
misaligned checkpoints loudly. Diagnosis path worth remembering: fp32
roundtrip through run.c proved the parser, then greedy (temp 0) comparison
isolated corruption to specific vocab rows → scale misalignment.

---

## Phase 5 — scaling up: real BPE + TinyStories

Phases 2–4 used a **byte-level** tokenizer (vocab 259): every character is its
own token, so "Once upon a time" is 16 tokens and the model spends most of its
capacity relearning English spelling. Phase 5 replaces that with a **byte-level
BPE** tokenizer trained from scratch, and trains a bigger model on a real
corpus.

### The tokenizer ([`src/bpe.py`](../src/bpe.py))

Byte-Pair Encoding starts from the 256 raw bytes and repeatedly fuses the most
frequent adjacent pair into a new token, until the vocabulary reaches a target
size (4096). The first merges it learns on TinyStories are exactly what you'd
guess: `he`, ` t`, ` a`, `in`, `the` — the statistical bones of English.

**The key insight: run.c needed zero changes.** Its `encode()` was already a
greedy "merge the highest-scored adjacent pair" loop — which *is* BPE decoding.
So we only had to (a) train the merges and (b) write them into the same
`tokenizer.bin` format with `score = -rank`, so the earliest-learned (highest
priority) merge wins each step. run.c then reproduces our exact tokenization.

Two conventions keep the Python trainer and the C encoder bit-identical:
- **Byte ids = `byte + 3`** — the same offset run.c's byte-fallback path uses,
  so unknown bytes map the same on both sides.
- **Leading-space pre-tokenization** (" the" is one unit) means no learned
  token ever spans a word boundary. So run.c's *global* greedy merge and our
  *per-word* greedy merge partition text identically — which also lets the
  Python encoder cache per unique word and stay fast.

Result: "Once upon a time, there was a little robot" → ~11 tokens instead of
~43. Each token now carries ~4 characters, so the model's fixed 256-token
context reaches ~1000 characters into a story.

### The data pipeline ([`src/prepare_data.py`](../src/prepare_data.py))

TinyStories (~300 MB of the train split used here) is tokenized **once**,
offline, into a flat `uint16` array on disk (vocab 4096 < 65536, so 2 bytes per
token). Training memory-maps that array and samples random 256-token windows —
the GPU never waits on Python. Each story is prefixed with BOS (id 1), so the
model learns BOS = "a new story starts", and generation from run.c (which
begins at BOS) opens like a fresh story. 314 MB of text → 78.7 M tokens.

### The model

Same architecture as before, scaled up and pointed at the bigger vocab:

```
dim=288  n_layers=6  n_heads=6  hidden_dim=768  vocab=4096  seq_len=256
~7.16M parameters   (vs 2.26M for Shakespeare)
```

The shape mirrors Karpathy's stories15M, but with a 4096-vocab instead of
32000 — most of stories15M's 15M params live in its huge embedding table;
ours spends them on the transformer blocks instead. Trained on Apple MPS with
AdamW + cosine LR; the training loop checkpoints the **val-best** weights so we
keep the best generalizer, not the last (most-overfit) step. One epoch ≈ 4800
iters, so a few thousand iters stays under a single pass — overfitting is a
non-issue at this data:param ratio, the opposite of the Shakespeare regime.

The payoff: real words, multi-sentence coherence, and simple narrative arcs
("Once upon a time… One day… but then… and they were happy") — the structure
TinyStories was designed to teach a small model.

---

## Phase 6 — instruction fine-tuning (`finetune.py`)

The TinyStories base model *continues* text. Phase 6 turns it into a model
that *follows an instruction* — the difference between a raw language model
and something you can actually prompt. This is **supervised fine-tuning
(SFT)**, the first stage of how real assistants are built.

Three techniques, none of which touch `run.c`:

### 1. A chat template (plain text)
Every SFT example is formatted as:
```
<s>User: <instruction>
Assistant: <response></s>
```
No new special tokens — `User:`/`Assistant:` are ordinary text the BPE
tokenizer already encodes, and BOS (id 1) is still the stop token `run.c`
halts on, so the model learns to end its turn. The demo wraps whatever you
type in this same template before handing it to the engine.

### 2. Loss masking (the one genuinely new idea)
We only want the model to learn to generate the **response**, not to parrot
the instruction. So when we build the `(input, target)` pair, every target
position that sits inside the prompt is set to `-100` — which is PyTorch's
`cross_entropy` `ignore_index`, silently dropped from the loss. The gradient
flows *only* from the answer tokens:
```python
full = prompt_ids + response_ids + [BOS]
x, y = full[:-1], full[1:]
for i in range(len(prompt_ids) - 1):
    y[i] = -100          # ignored by cross_entropy
```
Because `model.py`'s forward already calls `F.cross_entropy` with the default
`ignore_index=-100`, this needed **zero model changes** — just masked targets.

### 3. SFT from the pretrained base (transfer learning)
We load `models/tinystories.pt` and keep training — we are *not* starting from
random weights. The base already writes fluent stories; fine-tuning only
teaches the instruction *format*. Hence a low LR (5e-4) and a few epochs over
~50k examples, versus thousands of steps of pretraining.

### The data is synthesized from the corpus
There's no external instruction dataset. `finetune.py` builds pairs *from the
stories themselves*: pull content words out of a story, and the instruction
becomes "write a story using these words," with the story as the target
answer. This keeps everything in-domain for the TinyStories tokenizer, and
makes the task **verifiable** — we can measure what fraction of requested
words actually appear in generations (the honest quality metric, reported at
the end of a run).

### What this is and isn't
Full-fine-tuning all 7M parameters is fine at this scale; real instruct models
use **LoRA/PEFT** (train a small adapter, freeze the rest) to avoid updating
billions of weights, and follow SFT with a preference-alignment stage
(**RLHF/DPO**). What's here is step one — where instruction-following actually
originates. At 7M params it has no facts, no reasoning, no cross-turn memory;
it reliably obeys the *format* and *theme* (ask for a robot, get a robot), but
injecting specific requested words is hit-or-miss — ~24% of random content
words land (common ones far more often than rare ones), measured on held-out
prompts. The point is that the mechanism is the same one that scales up to real
assistants.
