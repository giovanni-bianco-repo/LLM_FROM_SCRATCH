import regex as re
from typing import List, Dict, Tuple, Iterable, Iterator

# Standard GPT-2 pre-tokenization regex
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_stats(ids: List[int], counts: Dict[Tuple[int, int], int] = None) -> Dict[Tuple[int, int], int]:
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_ids(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def _split_text_with_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """Split text into chunks, separating out special tokens."""
    if special_tokens:
        # Sort by length descending to ensure longest match wins
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(t) for t in special_tokens_sorted) + ")"
        return re.split(pattern, text)
    return [text]

def _pretokenize_text(text_chunks: List[str], special_tokens: List[str]) -> List[List[int]]:
    """Pre-tokenize text chunks into lists of byte IDs, skipping special tokens."""
    words: List[List[int]] = []
    compiled_pat = re.compile(GPT2_SPLIT_PATTERN)
    
    for chunk in text_chunks:
        if chunk in special_tokens:
            continue
        found = compiled_pat.findall(chunk)
        for token_str in found:
            words.append(list(token_str.encode("utf-8")))
    
    return words

def _build_initial_stats(words: List[List[int]]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Dict[int, int]]]:
    """Build initial pair statistics and indices from pre-tokenized words."""
    stats = {}
    indices = {}
    
    for i, word in enumerate(words):
        for j in range(len(word) - 1):
            pair = (word[j], word[j+1])
            stats[pair] = stats.get(pair, 0) + 1
            if pair not in indices:
                indices[pair] = {}
            indices[pair][i] = indices[pair].get(i, 0) + 1
    
    return stats, indices

def _update_left_neighbor(new_word: List[int], word: List[int], i: int, pair: Tuple[int, int], 
                          word_idx: int, stats: Dict, indices: Dict) -> None:
    """Update stats when breaking the left neighbor pair during a merge."""
    if len(new_word) > 0:
        prev_token = new_word[-1]
        old_pair_left = (prev_token, word[i])
        
        if old_pair_left in stats:
            stats[old_pair_left] -= 1
            if stats[old_pair_left] == 0:
                del stats[old_pair_left]
            
            if old_pair_left in indices and word_idx in indices[old_pair_left]:
                indices[old_pair_left][word_idx] -= 1
                if indices[old_pair_left][word_idx] == 0:
                    del indices[old_pair_left][word_idx]

def _update_right_neighbor(word: List[int], i: int, pair: Tuple[int, int], 
                           word_idx: int, stats: Dict, indices: Dict) -> None:
    """Update stats when breaking the right neighbor pair during a merge."""
    if i + 2 < len(word):
        next_token = word[i+2]
        old_pair_right = (word[i+1], next_token)
        
        # Only decrement if it's not the pair we are currently mass-deleting
        if old_pair_right != pair:
            if old_pair_right in stats:
                stats[old_pair_right] -= 1
                if stats[old_pair_right] == 0:
                    del stats[old_pair_right]
                
                if old_pair_right in indices and word_idx in indices[old_pair_right]:
                    indices[old_pair_right][word_idx] -= 1
                    if indices[old_pair_right][word_idx] == 0:
                        del indices[old_pair_right][word_idx]

def _add_new_neighbors(new_word: List[int], word: List[int], i: int, idx: int, 
                       word_idx: int, stats: Dict, indices: Dict) -> None:
    """Add new pair statistics after performing a merge."""
    # New Left Pair: (new_word[-2], idx)
    if len(new_word) > 1:
        prev = new_word[-2]
        new_pair_left = (prev, idx)
        stats[new_pair_left] = stats.get(new_pair_left, 0) + 1
        if new_pair_left not in indices:
            indices[new_pair_left] = {}
        indices[new_pair_left][word_idx] = indices[new_pair_left].get(word_idx, 0) + 1
    
    # New Right Pair: (idx, word[i+2])
    if i + 2 < len(word):
        next_token = word[i+2]
        new_pair_right = (idx, next_token)
        stats[new_pair_right] = stats.get(new_pair_right, 0) + 1
        if new_pair_right not in indices:
            indices[new_pair_right] = {}
        indices[new_pair_right][word_idx] = indices[new_pair_right].get(word_idx, 0) + 1

def _merge_pair_in_word(word: List[int], pair: Tuple[int, int], idx: int, 
                        word_idx: int, stats: Dict, indices: Dict) -> List[int]:
    """Merge all occurrences of a pair in a single word, updating statistics."""
    i = 0
    new_word = []
    
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
            # Match Found - update neighbors
            _update_left_neighbor(new_word, word, i, pair, word_idx, stats, indices)
            _update_right_neighbor(word, i, pair, word_idx, stats, indices)
            
            # Perform Merge
            new_word.append(idx)
            
            # Add New Neighbors
            _add_new_neighbors(new_word, word, i, idx, word_idx, stats, indices)
            
            i += 2
        else:
            # No Match
            new_word.append(word[i])
            i += 1
    
    return new_word

def _perform_merges(words: List[List[int]], vocab: Dict[int, bytes], stats: Dict, 
                    indices: Dict, num_merges: int) -> List[Tuple[bytes, bytes]]:
    """Perform BPE merges iteratively, returning the list of merges."""
    merges = []
    
    for _ in range(num_merges):
        if not stats:
            break
        
        # Tie-breaker: Max count, then Lexicographically greater PAIR of bytes
        pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]]))
        
        # Create new token
        idx = 256 + len(merges)
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        
        # Merge in words
        word_indices_to_update = indices.get(pair, {})
        del stats[pair]
        del indices[pair]
        
        for word_idx in word_indices_to_update:
            words[word_idx] = _merge_pair_in_word(words[word_idx], pair, idx, word_idx, stats, indices)
    
    return merges

def _add_special_tokens_to_vocab(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], 
                                  special_tokens: List[str]) -> None:
    """Add special tokens to the vocabulary."""
    current_id = 256 + len(merges)
    for st in special_tokens:
        vocab[current_id] = st.encode("utf-8")
        current_id += 1

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the input text file."""
    # Read Input
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Setup Vocab (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    
    # Handle Special Tokens and Pre-tokenize
    text_chunks = _split_text_with_special_tokens(text, special_tokens)
    words = _pretokenize_text(text_chunks, special_tokens)
    
    # Build Initial Stats & Index
    stats, indices = _build_initial_stats(words)
    
    # Perform Iterative Merging
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = _perform_merges(words, vocab, stats, indices, num_merges)
    
    # Add Special Tokens
    _add_special_tokens_to_vocab(vocab, merges, special_tokens)
    
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab_inverse = {v: k for k, v in vocab.items()}
        
        # Build merge map: (byte_seq_1, byte_seq_2) -> (rank, new_id)
        self.merge_map = {}
        for rank, (b1, b2) in enumerate(merges):
            combined = b1 + b2
            if combined in self.vocab_inverse:
                new_id = self.vocab_inverse[combined]
                self.merge_map[(b1, b2)] = (rank, new_id)

    def encode(self, text: str) -> List[int]:
        if self.special_tokens:
            self.special_tokens.sort(key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]
            
        ids = []
        compiled_pat = re.compile(GPT2_SPLIT_PATTERN)

        for chunk in chunks:
            if not chunk: continue
            
            if chunk in self.special_tokens:
                ids.append(self.vocab_inverse[chunk.encode("utf-8")])
                continue
            
            pre_tokens = compiled_pat.findall(chunk)
            
            for token_str in pre_tokens:
                token_ids = [self.vocab_inverse[bytes([b])] for b in token_str.encode("utf-8")]
                
                while len(token_ids) >= 2:
                    stats = get_stats(token_ids)
                    
                    # Find the pair with the lowest rank (earliest merge)
                    best_pair = None
                    min_rank = float('inf')
                    target_id = -1
                    
                    for pair in stats:
                        p_bytes = (self.vocab[pair[0]], self.vocab[pair[1]])
                        if p_bytes in self.merge_map:
                            rank, new_id = self.merge_map[p_bytes]
                            if rank < min_rank:
                                min_rank = rank
                                best_pair = pair
                                target_id = new_id
                    
                    if best_pair is None:
                        break
                        
                    token_ids = merge_ids(token_ids, best_pair, target_id)
                ids.extend(token_ids)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: List[int]) -> str:
        byte_seq = b"".join([self.vocab[i] for i in ids])
        return byte_seq.decode("utf-8", errors="replace")
