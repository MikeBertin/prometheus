"""
PROMETHEUS — model.py

The Llama-2 style transformer in PyTorch — the *training-time twin* of
src/run.c. Every architectural choice here mirrors the C forward pass exactly,
because the whole point is that weights trained here run unmodified there:

    RMSNorm  eps = 1e-5           (run.c rmsnorm)
    RoPE     base 10000, rotating ADJACENT dim pairs (i, i+1) per head
    Attention scaled by 1/sqrt(head_size), causal
    SwiGLU   w2( silu(w1 x) * (w3 x) )
    Classifier tied to the embedding table (=> positive vocab_size in header)

Anything that exists only at training time (dropout, autograd, the loss) has
no counterpart in run.c and doesn't need one.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 192
    n_layers: int = 5
    n_heads: int = 6
    n_kv_heads: int = 6          # == n_heads here (no GQA at this scale)
    vocab_size: int = 259        # byte-level: 3 specials + 256 bytes
    hidden_dim: int = 512        # SwiGLU inner width
    max_seq_len: int = 256
    norm_eps: float = 1e-5       # MUST match run.c's rmsnorm
    dropout: float = 0.0


class RMSNorm(nn.Module):
    """x / rms(x) * weight — identical math to run.c rmsnorm()."""
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


# ---------------------------------------------------------------------------
# RoPE. We precompute the complex rotations e^{i * pos * freq} and multiply
# each ADJACENT pair (x[2k], x[2k+1]) viewed as a complex number. This is the
# same pairing run.c uses in its (i, i+1) loop — NOT the "rotate half"
# convention used by HF Llama, which pairs (x[k], x[k + d/2]).
# ---------------------------------------------------------------------------
def precompute_freqs_cis(head_size: int, seq_len: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, head_size, 2)[: head_size // 2].float() / head_size))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)                     # (seq_len, head_size/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq, xk, freqs_cis):
    # (B, T, H, head_size) -> complex (B, T, H, head_size/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[: xq.shape[1]].view(1, xq.shape[1], 1, -1)  # broadcast over B, H
    xq_out = torch.view_as_real(xq_ * fc).flatten(3)
    xk_out = torch.view_as_real(xk_ * fc).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x, n_rep: int):
    """Expand kv heads to match query heads (GQA). Identity when n_rep == 1."""
    if n_rep == 1:
        return x
    b, t, kvh, hs = x.shape
    return x[:, :, :, None, :].expand(b, t, kvh, n_rep, hs).reshape(b, t, kvh * n_rep, hs)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = args.n_heads // args.n_kv_heads
        self.head_size = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_size, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_size, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_size, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_size, args.dim, bias=False)
        self.dropout = args.dropout

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_size)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_size)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_size)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        xk, xv = repeat_kv(xk, self.n_rep), repeat_kv(xv, self.n_rep)

        # (B, H, T, head_size); SDPA's causal mask == run.c's t <= pos loop,
        # and its default scale is 1/sqrt(head_size), same as run.c.
        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
        out = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU: w2( silu(w1 x) * (w3 x) ) — run.c's hb/hb2 loop."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)  # down
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.attention = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.feed_forward = FeedForward(args)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)   # residual
        x = x + self.feed_forward(self.ffn_norm(x))                 # residual
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(TransformerBlock(args) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # Weight tying: classifier IS the embedding table. run.c reads the
        # sign of vocab_size in the header to know this (positive = tied).
        self.output.weight = self.tok_embeddings.weight

        freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init on the residual-writing projections, so the
        # residual stream's variance stays ~constant with depth at init.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        x = self.dropout(self.tok_embeddings(tokens))
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.norm(x)
        logits = self.output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Minimal sampler for training-time sanity checks (the real sampler
        for actual use is in run.c)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.args.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        self.train()
        return idx
