import os
from typing import Dict, List, Tuple, Optional
import regex as re

# ---------- Pre-tokenization (GPT-2 style) ----------

PAT: str = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| """
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pre_tokenize(text: str) -> Dict[Tuple[bytes, ...], int]:
    """
    Split text into GPT-2-like tokens, encode each token to UTF-8,
    then represent each token as a tuple of ONE-BYTE bytes objects.
    Return frequency counts of those byte-symbol sequences.
    """
    freq: Dict[Tuple[bytes, ...], int] = {}
    for m in re.finditer(PAT, text):
        tok: str = m.group(0)
        b: bytes = tok.encode("utf-8")
        # b'Hello' -> (b'H', b'e', b'l', b'l', b'o')
        seq: Tuple[bytes, ...] = tuple(b[i : i + 1] for i in range(len(b)))
        freq[seq] = freq.get(seq, 0) + 1
    return freq


# ---------- Vocab helpers ----------


def add_special_tokens(
    vocab: Dict[int, bytes], special_tokens: Optional[List[str]] = None
) -> Dict[int, bytes]:
    """
    Append special tokens (as UTF-8 bytes) to the vocab dict.
    Keys are integer IDs; values are the byte string for that symbol.
    """
    if special_tokens is None:
        special_tokens = []
    next_id: int = len(vocab)
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    return vocab


# ---------- BPE training core ----------


def _pair_stats(freqs: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    """
    Count adjacent pair frequencies across all token sequences,
    weighted by the sequence frequency.
    """
    stats: Dict[Tuple[bytes, bytes], int] = {}
    for token, count in freqs.items():
        if len(token) < 2:
            continue
        for a, b in zip(token, token[1:]):
            stats[(a, b)] = stats.get((a, b), 0) + count
    return stats


def _merge_once(
    freqs: Dict[Tuple[bytes, ...], int], pair: Tuple[bytes, bytes]
) -> Dict[Tuple[bytes, ...], int]:
    """
    Replace all non-overlapping occurrences of `pair` in each token tuple
    with the merged symbol (a+b). Each element of a token is a bytes object.
    """
    merged_symbol: bytes = pair[0] + pair[1]
    new_freqs: Dict[Tuple[bytes, ...], int] = {}

    for token, count in freqs.items():
        i: int = 0
        new_token: List[bytes] = [] #the new token initialised as a list to be mutable
        n: int = len(token)
        while i < n:
            #check if it an occurrence of the tuple: if yes add the merged and increment by 2, else add each elem
            if i + 1 < n and token[i] == pair[0] and token[i + 1] == pair[1]:
                new_token.append(merged_symbol) 
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_token: Tuple[bytes, ...] = tuple(new_token)
        new_freqs[new_token] = new_freqs.get(new_token, 0) + count

    return new_freqs


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: Optional[List[str]] = None,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-pair encoding vocabulary from a text file.

    Returns:
        vocab:  id -> bytes symbol (single or merged)
        merges: sequence of merges as (lhs_bytes, rhs_bytes)
    """
    # 1) initialize vocab with raw bytes
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 2) read and pre-tokenize
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text: str = f.read()
    freqs: Dict[Tuple[bytes, ...], int] = pre_tokenize(text)

    merges: List[Tuple[bytes, bytes]] = []
    next_id: int = len(vocab)

    # ensure target includes at least the 256 base bytes
    target_vocab: int = max(vocab_size, 256)

    # 3) iterative merges (deterministic tie-break by (count, pair))
    while len(vocab) < target_vocab:
        stats: Dict[Tuple[bytes, bytes], int] = _pair_stats(freqs)
        if not stats:
            break
        best_pair: Tuple[bytes, bytes] = max(
            stats.items(), key=lambda kv: (kv[1], kv[0])
        )[0]

        # apply the merge
        freqs = _merge_once(freqs, best_pair)
        merges.append(best_pair)

        # add merged symbol into vocab
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1

    # 4) add special tokens last so merges never include them
    vocab = add_special_tokens(vocab, special_tokens)

    return vocab, merges


if __name__ == "__main__":
    # tiny smoke test: trains a few merges on this file itself
    this_path: str = __file__
    vocab_out, merges_out = train_bpe(
        this_path, vocab_size=300, special_tokens=["<pad>", "<eos>"]
    )
    print(f"vocab size: {len(vocab_out)}; merges learned: {len(merges_out)}")
