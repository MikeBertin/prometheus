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
