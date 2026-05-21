"""Build compressed WordPiece vocab using a WordPiece-native zstd dictionary.

Trains a zstd dictionary from the 30K WordPiece token strings themselves,
so common substrings (play, ##ing, etc.) are captured as match entries.
Compresses each token against this dictionary, strips zstd frame overhead,
and stores the content bytes padded to uniform length with sentinel 0x00.

Output:
  vocab/wp_dict.zdict           — zstd dictionary trained on token strings
  vocab/compressed_vocab.pt     — [vocab_size, max_content_bytes] uint8 tensor
"""

import os, sys, pickle
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer
import zstandard as zstd


SENTINEL = 0


def token_text(token_id: int, token_str: str) -> str:
    if token_str.startswith("##"):
        return token_str[2:]
    if token_str.startswith("[") and token_str.endswith("]"):
        return token_str[1:-1].lower()
    if token_str.startswith("unused"):
        return "unused"
    return token_str


def strip_zstd_frame(compressed: bytes, frame_offset: int = 4) -> bytes:
    """Strip zstd magic + frame header, returning just the content blocks.

    zstd frame layout:
      bytes 0-3:   magic number (0xFD2FB528)
      bytes 4+:    frame header (varies, includes dict ID if used)
      then:        block(s), each with 3-byte block header + content

    We strip everything up to the first block header, then parse block
    headers to extract content bytes from all blocks.
    """
    # Only handle single-block frames (level 3 with small inputs)
    pos = frame_offset  # skip magic
    # Read frame header
    header_desc = compressed[pos]
    pos += 1

    if header_desc & 0x08:  # dictionary flag
        dict_id_flag = (header_desc >> 2) & 0x03
        if dict_id_flag == 0:
            pos += 1
        elif dict_id_flag == 1:
            pos += 2
        else:
            pos += 4

    if header_desc & 0x40:  # content size flag
        content_size_flag = header_desc & 0x07
        if content_size_flag == 0:
            pass  # no content size
        elif content_size_flag == 1:
            pos += 1
        elif content_size_flag == 2:
            pos += 2
        elif content_size_flag == 4:
            pos += 4
        elif content_size_flag == 5:
            pos += 8

    # Now pos points to the first block header
    content_parts = []
    while pos < len(compressed):
        if pos + 3 > len(compressed):
            break
        block_header = int.from_bytes(compressed[pos:pos+3], 'little')
        last_block = block_header >> 7
        block_type = (block_header >> 5) & 0x03
        block_size = block_header & 0x7FF  # 11-bit raw size
        pos += 3

        if block_type == 0:  # raw block
            content_parts.append(compressed[pos:pos+block_size])
            pos += block_size
        elif block_type == 1:  # RLE block
            content_parts.append(compressed[pos:pos+1] * block_size)
            pos += 1
        elif block_type == 2:  # compressed block
            content_parts.append(compressed[pos:pos+block_size])
            pos += block_size
        else:
            break  # reserved

        if last_block:
            break

    return b''.join(content_parts)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "vocab")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    items = sorted(vocab.items(), key=lambda x: x[1])
    print(f"Vocab size: {vocab_size}")

    # Collect all token texts as training samples for the dictionary
    print("Collecting training samples...")
    samples = []
    for token_str, token_id in items:
        text = token_text(token_id, token_str)
        samples.append(text.encode("utf-8"))

    # Train a zstd dictionary from WordPiece tokens
    dict_path = os.path.join(out_dir, "wp_dict.zdict")
    if not os.path.exists(dict_path):
        print("Training WordPiece-native zstd dictionary...")
        dict_data = zstd.train_dictionary(
            4096,             # dict_size — 4KB
            samples,
        )
        with open(dict_path, "wb") as f:
            f.write(dict_data.as_bytes())
        print(f"Dictionary saved to {dict_path} ({len(dict_data.as_bytes())} bytes)")
    else:
        print(f"Dictionary already exists at {dict_path}, loading...")
        with open(dict_path, "rb") as f:
            dict_data = zstd.ZstdCompressionDict(f.read())

    # Compress each token with framing stripped
    cctx = zstd.ZstdCompressor(level=3, dict_data=dict_data,
                                write_content_size=False,
                                write_checksum=False)

    compressed = []
    max_len = 0

    print("Compressing tokens...")
    for token_str, token_id in items:
        text = token_text(token_id, token_str)
        text_bytes = text.encode("utf-8")
        full_frame = cctx.compress(text_bytes)

        # Strip zstd framing to get just the content bytes
        content = strip_zstd_frame(full_frame)
        compressed.append(content)
        if len(content) > max_len:
            max_len = len(content)

    print(f"Max content bytes: {max_len}")

    # Pad and build tensor
    tensor = torch.full((vocab_size, max_len + 1), SENTINEL, dtype=torch.uint8)
    for token_id, content in enumerate(compressed):
        tensor[token_id, :len(content)] = torch.tensor(list(content), dtype=torch.uint8)

    print(f"Tensor shape: {tensor.shape}")
    print(f"Sentinels: {(tensor == SENTINEL).sum().item() / tensor.numel() * 100:.1f}% of entries")
    print(f"Mean bytes per token: {(tensor != SENTINEL).sum(dim=1).float().mean().item():.2f}")

    out_path = os.path.join(out_dir, "compressed_vocab.pt")
    torch.save(tensor, out_path)
    meta = {
        "vocab_size": vocab_size,
        "max_token_bytes": max_len + 1,
        "sentinel_byte": SENTINEL,
        "source": "bert-base-uncased",
        "dict_path": dict_path,
    }
    meta_path = os.path.join(out_dir, "compressed_vocab_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {out_path}")
    print(f"Meta: {meta_path}")

    # Quick stats
    byte_counts = (tensor != SENTINEL).sum(dim=1)
    print(f"\nCompression stats:")
    print(f"  25th percentile: {byte_counts.float().quantile(0.25).item():.0f} bytes")
    print(f"  50th percentile: {byte_counts.float().quantile(0.50).item():.0f} bytes")
    print(f"  75th percentile: {byte_counts.float().quantile(0.75).item():.0f} bytes")
    print(f"  90th percentile: {byte_counts.float().quantile(0.90).item():.0f} bytes")
    print(f"  99th percentile: {byte_counts.float().quantile(0.99).item():.0f} bytes")

    # Show sample tokens
    print(f"\nSample tokens:")
    for name, tid in [("the", 1996), ("cat", 4937), ("superfluous", 3565),
                       ("pterodactyl", 23921)]:
        c = compressed[tid]
        text = token_text(tid, tokenizer.convert_ids_to_tokens(tid))
        orig_len = len(text.encode("utf-8"))
        print(f"  {name:>15s} (ID {tid:>5d}): orig={orig_len:>2d}B → content={len(c):>2d}B  ratio={orig_len/max(len(c),1):.2f}x")


if __name__ == "__main__":
    main()
