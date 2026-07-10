"""
PROMETHEUS — export.py

Serializes a trained PyTorch Transformer into the flat float32 binary that
run.c's read_checkpoint() mmaps. THE ORDER BELOW IS THE CONTRACT — it must
match memory_map_weights() in src/run.c field for field:

    header: 7 x int32  (dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                        vocab_size, seq_len)
            vocab_size is written POSITIVE because our classifier is tied to
            the embedding table (negative would mean "untied wcls follows").
    token_embedding_table   (vocab, dim)
    rms_att_weight          per layer, concatenated
    wq, wk, wv, wo          each: per layer, concatenated
    rms_ffn_weight          per layer
    w1, w2, w3              each: per layer
    rms_final_weight        (dim,)
    freq_cis_real/imag      legacy RoPE tables run.c SKIPS over — we write
                            zeros of the right size just to keep the offsets
                            aligned (seq_len * head_size/2 floats, twice)

nn.Linear stores weight as (out_features, in_features) row-major, which is
exactly the (d, n) row-major layout run.c's matmul indexes — so we can dump
each weight tensor as-is, no transpose.
"""
import struct
import sys

import torch

from model import Transformer, ModelArgs


def serialize_fp32(f, tensor):
    f.write(tensor.detach().cpu().reshape(-1).to(torch.float32).numpy().tobytes())


def legacy_export(model: Transformer, path: str) -> None:
    a = model.args
    head_size = a.dim // a.n_heads
    with open(path, "wb") as f:
        # header — vocab_size positive => classifier tied to embeddings
        f.write(struct.pack("<7i", a.dim, a.hidden_dim, a.n_layers,
                            a.n_heads, a.n_kv_heads, a.vocab_size, a.max_seq_len))

        serialize_fp32(f, model.tok_embeddings.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.attention_norm.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wq.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wk.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wv.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wo.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.ffn_norm.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.feed_forward.w1.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.feed_forward.w2.weight)
        for layer in model.layers:
            serialize_fp32(f, layer.feed_forward.w3.weight)
        serialize_fp32(f, model.norm.weight)

        # legacy freq_cis blocks (run.c skips them; zeros keep offsets right)
        pad = torch.zeros(a.max_seq_len * head_size // 2)
        serialize_fp32(f, pad)
        serialize_fp32(f, pad)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"wrote {path} ({n_params:,} params)")


def load_checkpoint(ckpt_path: str) -> Transformer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = Transformer(ModelArgs(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python export.py <ckpt.pt> <out.bin>")
        sys.exit(1)
    legacy_export(load_checkpoint(sys.argv[1]), sys.argv[2])
