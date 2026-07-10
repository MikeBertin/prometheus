"""
PROMETHEUS — export.py

Serializes trained weights into the binaries the C engines mmap:

  legacy (v1) fp32  -> run.c    flat float32, 28-byte header
  q80    (v2) int8  -> runq.c   "ak42" 256-byte header, group-wise Q8_0

Inputs: either a training checkpoint (ckpt.pt) or an existing legacy fp32
.bin — the latter means any llama2.c-format model (e.g. Karpathy's
stories15M.bin) can be quantized without its original PyTorch checkpoint.

  python export.py models/ckpt.pt        models/shakespeare.bin
  python export.py models/ckpt.pt        models/shakespeare_q80.bin --q80
  python export.py models/stories15M.bin models/stories15M_q80.bin  --q80

THE ORDER OF TENSORS IS THE CONTRACT — it must match memory_map_weights()
in run.c (legacy) / runq.c (q80) field for field.

Q8_0 quantization: split each tensor into groups of --gs values; per group
scale = max|w| / 127, q = round(w / scale) as int8. Symmetric (no zero
point), group-wise (one outlier only hurts its own group). We report the
worst per-tensor reconstruction error so you can see what int8 costs.
"""
import argparse
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# weight loading — both sources produce the same plain-numpy structure:
#   cfg: dict(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
#   t:   dict with 'emb', 'final_norm', 'wcls' (None if tied) and per-layer
#        lists 'att_norm','ffn_norm','wq','wk','wv','wo','w1','w2','w3'
# ---------------------------------------------------------------------------

def load_from_ckpt(path):
    import torch
    from model import Transformer, ModelArgs
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(ModelArgs(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    a = model.args
    cfg = dict(dim=a.dim, hidden_dim=a.hidden_dim, n_layers=a.n_layers,
               n_heads=a.n_heads, n_kv_heads=a.n_kv_heads,
               vocab_size=a.vocab_size, seq_len=a.max_seq_len)
    f32 = lambda t: t.detach().cpu().reshape(-1).to(torch.float32).numpy()
    t = {
        "emb": f32(model.tok_embeddings.weight),
        "att_norm": [f32(l.attention_norm.weight) for l in model.layers],
        "ffn_norm": [f32(l.ffn_norm.weight) for l in model.layers],
        "wq": [f32(l.attention.wq.weight) for l in model.layers],
        "wk": [f32(l.attention.wk.weight) for l in model.layers],
        "wv": [f32(l.attention.wv.weight) for l in model.layers],
        "wo": [f32(l.attention.wo.weight) for l in model.layers],
        "w1": [f32(l.feed_forward.w1.weight) for l in model.layers],
        "w2": [f32(l.feed_forward.w2.weight) for l in model.layers],
        "w3": [f32(l.feed_forward.w3.weight) for l in model.layers],
        "final_norm": f32(model.norm.weight),
        "wcls": None,  # our models tie the classifier to the embedding
    }
    return cfg, t


def load_from_legacy_bin(path):
    """Parse a llama2.c legacy fp32 checkpoint back into tensors."""
    raw = Path(path).read_bytes()
    dim, hidden, L, H, KVH, vocab, seq = struct.unpack("<7i", raw[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    cfg = dict(dim=dim, hidden_dim=hidden, n_layers=L, n_heads=H,
               n_kv_heads=KVH, vocab_size=vocab, seq_len=seq)
    head_size = dim // H
    kv_dim = dim * KVH // H

    data = np.frombuffer(raw, dtype=np.float32, offset=28)
    pos = 0
    def take(n):
        nonlocal pos
        out = data[pos:pos + n]; pos += n
        return out
    def take_layers(n_each):
        return [take(n_each) for _ in range(L)]

    t = {}
    t["emb"] = take(vocab * dim)
    t["att_norm"] = take_layers(dim)
    t["wq"] = take_layers(dim * dim)
    t["wk"] = take_layers(dim * kv_dim)
    t["wv"] = take_layers(dim * kv_dim)
    t["wo"] = take_layers(dim * dim)
    t["ffn_norm"] = take_layers(dim)
    t["w1"] = take_layers(dim * hidden)
    t["w2"] = take_layers(hidden * dim)
    t["w3"] = take_layers(dim * hidden)
    t["final_norm"] = take(dim)
    take(seq * head_size // 2); take(seq * head_size // 2)  # legacy RoPE tables
    t["wcls"] = None if shared else take(vocab * dim)
    return cfg, t


# ---------------------------------------------------------------------------
# legacy (v1) fp32 writer — run.c's format
# ---------------------------------------------------------------------------

def write_f32(f, arr):
    f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def legacy_export_tensors(cfg, t, path):
    head_size = cfg["dim"] // cfg["n_heads"]
    shared = t["wcls"] is None
    with open(path, "wb") as f:
        vocab = cfg["vocab_size"] if shared else -cfg["vocab_size"]
        f.write(struct.pack("<7i", cfg["dim"], cfg["hidden_dim"], cfg["n_layers"],
                            cfg["n_heads"], cfg["n_kv_heads"], vocab, cfg["seq_len"]))
        write_f32(f, t["emb"])
        for name in ("att_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3"):
            for layer_tensor in t[name]:
                write_f32(f, layer_tensor)
        write_f32(f, t["final_norm"])
        pad = np.zeros(cfg["seq_len"] * head_size // 2, dtype=np.float32)
        write_f32(f, pad); write_f32(f, pad)
        if not shared:
            write_f32(f, t["wcls"])
    print(f"wrote {path} (legacy fp32)")


def legacy_export(model, path):
    """Back-compat entry point used by train.py."""
    import torch  # noqa: F401  (loaded lazily elsewhere)
    cfg, t = _tensors_from_live_model(model)
    legacy_export_tensors(cfg, t, path)


def _tensors_from_live_model(model):
    import torch
    a = model.args
    cfg = dict(dim=a.dim, hidden_dim=a.hidden_dim, n_layers=a.n_layers,
               n_heads=a.n_heads, n_kv_heads=a.n_kv_heads,
               vocab_size=a.vocab_size, seq_len=a.max_seq_len)
    f32 = lambda t: t.detach().cpu().reshape(-1).to(torch.float32).numpy()
    return cfg, {
        "emb": f32(model.tok_embeddings.weight),
        "att_norm": [f32(l.attention_norm.weight) for l in model.layers],
        "ffn_norm": [f32(l.ffn_norm.weight) for l in model.layers],
        "wq": [f32(l.attention.wq.weight) for l in model.layers],
        "wk": [f32(l.attention.wk.weight) for l in model.layers],
        "wv": [f32(l.attention.wv.weight) for l in model.layers],
        "wo": [f32(l.attention.wo.weight) for l in model.layers],
        "w1": [f32(l.feed_forward.w1.weight) for l in model.layers],
        "w2": [f32(l.feed_forward.w2.weight) for l in model.layers],
        "w3": [f32(l.feed_forward.w3.weight) for l in model.layers],
        "final_norm": f32(model.norm.weight),
        "wcls": None,
    }


# ---------------------------------------------------------------------------
# q80 (v2) int8 writer — runq.c's format
# ---------------------------------------------------------------------------

def q80_quantize(w, gs):
    """Group-wise symmetric int8. Returns (int8 values, fp32 scales, max_err)."""
    assert w.size % gs == 0, f"tensor size {w.size} not divisible by group size {gs}"
    groups = w.reshape(-1, gs).astype(np.float32)
    scale = np.abs(groups).max(axis=1) / 127.0
    scale[scale == 0.0] = 1.0                       # all-zero group
    q = np.round(groups / scale[:, None]).astype(np.int8)
    err = np.abs(q.astype(np.float32) * scale[:, None] - groups).max()
    return q.reshape(-1), scale.astype(np.float32), float(err)


def q80_export_tensors(cfg, t, path, gs=64):
    # Groups must align with matrix ROWS: runq.c's matmul indexes a weight's
    # scales as (row*n + j)/GS, which is only right if GS divides every row
    # length (dim and hidden_dim). Shrink GS until it does — Karpathy's
    # exporter does the same. (dim=288 models silently corrupt at GS=64:
    # every odd row gets its neighbour's scales.)
    while cfg["dim"] % gs != 0 or cfg["hidden_dim"] % gs != 0:
        gs //= 2
        assert gs >= 1, "no valid group size divides dim and hidden_dim"
    shared = t["wcls"] is None
    max_err = 0.0

    def write_q(f, arr, label):
        nonlocal max_err
        q, s, err = q80_quantize(np.asarray(arr), gs)
        f.write(q.tobytes())
        f.write(s.tobytes())
        max_err = max(max_err, err)

    with open(path, "wb") as f:
        # 256-byte header: magic "ak42", version 2, config, flags
        f.write(struct.pack("<I", 0x616B3432))
        f.write(struct.pack("<i", 2))
        f.write(struct.pack("<7i", cfg["dim"], cfg["hidden_dim"], cfg["n_layers"],
                            cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"],
                            cfg["seq_len"]))
        f.write(struct.pack("<B", 1 if shared else 0))
        f.write(struct.pack("<i", gs))
        f.write(b"\0" * (256 - f.tell()))

        # fp32 norms first (they stay full precision in runq.c)
        for arr in t["att_norm"]: write_f32(f, arr)
        for arr in t["ffn_norm"]: write_f32(f, arr)
        write_f32(f, t["final_norm"])

        # then every matmul weight, quantized — order == runq.c's walk
        write_q(f, t["emb"], "emb")
        for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            for i, arr in enumerate(t[name]):
                write_q(f, arr, f"{name}[{i}]")
        if not shared:
            write_q(f, t["wcls"], "wcls")

    size = Path(path).stat().st_size
    print(f"wrote {path} (q80, GS={gs}, {size/1e6:.1f} MB, "
          f"worst reconstruction error {max_err:.5f})")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ckpt.pt (training checkpoint) or legacy fp32 .bin")
    ap.add_argument("output")
    ap.add_argument("--q80", action="store_true", help="int8 Q8_0 (v2) for runq.c")
    ap.add_argument("--gs", type=int, default=64, help="quantization group size")
    args = ap.parse_args()

    if args.input.endswith(".pt"):
        cfg, t = load_from_ckpt(args.input)
    else:
        cfg, t = load_from_legacy_bin(args.input)

    if args.q80:
        q80_export_tensors(cfg, t, args.output, gs=args.gs)
    else:
        legacy_export_tensors(cfg, t, args.output)
