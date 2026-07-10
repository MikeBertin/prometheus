"""
PROMETHEUS — tokenizer_export.py

Writes a *byte-level* tokenizer in the exact binary format run.c's
build_tokenizer() reads:

    int32  max_token_length
    then, for each of vocab_size tokens:
        float32 score      (BPE merge priority — all 0 here, no merges)
        int32   length     (bytes of the token string)
        bytes   the token string

Vocabulary layout (matches the Llama convention run.c assumes):
    id 0        <unk>
    id 1        <s>     (BOS — generate() starts from it, and stops on it)
    id 2        </s>    (EOS)
    id 3 + b    one token per raw byte b in 0..255

For printable ASCII (and \t \n \r) the token string IS the raw character, so
run.c's encode() finds it by binary search and decode() prints it directly.
Everything else is spelled "<0xNN>", which decode() already special-cases via
sscanf into a raw byte. Net effect: run.c needs ZERO changes to speak this
tokenizer — encoding is one token per byte, and the BPE merge loop simply
finds nothing to merge.

Why byte-level? It removes tokenizer *training* from the pipeline entirely
(nothing to learn — the vocab is the byte alphabet), so Phase 2 can focus on
the model. The price is longer sequences: 1 byte = 1 token.
"""
import struct
import sys
import os

VOCAB_SIZE = 259  # 3 specials + 256 bytes


def token_string(b: int) -> bytes:
    """The vocab string for raw byte b."""
    if 32 <= b <= 126 or b in (9, 10, 13):  # printable ASCII + tab/newline/CR
        return bytes([b])
    return f"<0x{b:02X}>".encode()


def export(path: str) -> None:
    tokens = [b"<unk>", b"<s>", b"</s>"] + [token_string(b) for b in range(256)]
    assert len(tokens) == VOCAB_SIZE
    max_len = max(len(t) for t in tokens)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", max_len))
        for t in tokens:
            f.write(struct.pack("<fi", 0.0, len(t)))  # score, length
            f.write(t)
    print(f"wrote {path}: vocab_size={VOCAB_SIZE}, max_token_length={max_len}, "
          f"{os.path.getsize(path)} bytes")


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "models/byte_tokenizer.bin")
