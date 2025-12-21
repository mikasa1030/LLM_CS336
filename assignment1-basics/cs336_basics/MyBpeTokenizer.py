from time import sleep
import regex as re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set,Optional,Iterable,Iterator
import heapq

from torch.onnx import select_model_mode_for_export # 修正导入

class Pretokenizer:
    def __init__(self, special_tokens: List[str]):
        self.special_tokens = {token: token.encode('utf-8') for token in special_tokens}
        self.word_pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens.keys(), key=len, reverse=True)
            special_pat_str = "|".join(re.escape(token) for token in sorted_special_tokens)
            self.special_pat = re.compile(f'({special_pat_str})')
        else:
            self.special_pat = None

    def tokenize(self, text: str) -> List[bytes]:
        if not self.special_pat:
            return [match.group(0).encode('utf-8') for match in self.word_pat.finditer(text)]
        
        all_tokens = []
        chunks = self.special_pat.split(text)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                all_tokens.append(self.special_tokens[chunk])
            else:
                word_tokens = [match.group(0).encode('utf-8') for match in self.word_pat.finditer(chunk)]
                all_tokens.extend(word_tokens)
        return all_tokens

class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: List[str]): # 类型修正 str -> List[str]
        if vocab_size < 256:
            raise ValueError("vocab_size is smaller than 256")

        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.pretokenizer = Pretokenizer(special_tokens)

        # ❌ 修正1：bytes([i]) 而不是 bytes(i)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        for token in self.special_tokens:
            encode_token = token.encode('utf-8')
            if encode_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = encode_token

    def tokenizer(self, text: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        all_tokens = self.pretokenizer.tokenize(text)
        special_tokens_set = set(self.pretokenizer.special_tokens.values()) # 重命名避免冲突
        word_freq = Counter(token for token in all_tokens if token not in special_tokens_set)

        splits: Dict[bytes, List[bytes]] = {token: [bytes([b]) for b in token] for token in word_freq}
        pair_freq: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        pair_words: Dict[Tuple[bytes, bytes], Set[bytes]] = defaultdict(set)
        freq_max_heap: List[Tuple[int, Tuple[bytes, bytes]]] = []

        # ❌ 修正2：加 .items()
        for word, freq in word_freq.items():
            word_split = splits[word]
            if len(word_split) > 1:
                for p1, p2 in zip(word_split[:-1], word_split[1:]):
                    pair = (p1, p2)
                    pair_freq[pair] += freq
                    pair_words[pair].add(word)
        
        # ❌ 修正4：直接用 imported heapq 或者 heappush
        for pair, freq in pair_freq.items(): # 这里也要加 .items()
            heapq.heappush(freq_max_heap, (-freq, pair))
        
        merged: List[Tuple[bytes, bytes]] = []
        num_merged_pair = self.vocab_size - len(self.vocab)
        
        while len(merged) < num_merged_pair:
            best_pair = self.find_best_pair(freq_max_heap, pair_freq)
            if best_pair is None:
                break

            merged.append(best_pair)
            new_token_bytes = best_pair[0] + best_pair[1]
            self.vocab[len(self.vocab)] = new_token_bytes

            self.update_data_struct(best_pair, new_token_bytes, splits, pair_freq, pair_words, word_freq, freq_max_heap)

        return self.vocab, merged

    def find_best_pair(self, freq_max_heap: List, pair_freq: Dict) -> Optional[Tuple[bytes, bytes]]:
        while freq_max_heap:
            neg_freq, pair = heapq.heappop(freq_max_heap)
            freq = -neg_freq

            if pair not in pair_freq or pair_freq[pair] != freq:
                continue
            
            best_pair = pair
            candidates = [] # ❌ 修正5：拼写修正
            
            while freq_max_heap and freq_max_heap[0][0] == neg_freq:
                _, other_pair = heapq.heappop(freq_max_heap)
                if other_pair in pair_freq and pair_freq[other_pair] == freq:
                    if other_pair > best_pair:
                        candidates.append(best_pair)
                        best_pair = other_pair
                    else:
                        candidates.append(other_pair)
            
            for p in candidates:
                heapq.heappush(freq_max_heap, (neg_freq, p))
            return best_pair
        return None

    # ❌ 修正3：加上 self
    def update_data_struct(self, best_pair: Tuple, new_token_bytes: bytes, splits: Dict, pair_freq: Dict, pair_words: Dict, word_freq: Dict, freq_max_heap: List):
        for word in list(pair_words.get(best_pair, [])):
            freq = word_freq[word]
            word_list = splits[word]

            i = 0
            while i < len(word_list) - 1: # 注意这里的边界条件通常是 len-1
                if word_list[i] == best_pair[0] and word_list[i+1] == best_pair[1]:
                    word_list[i] = new_token_bytes
                    word_list.pop(i+1)
                    if i>0:
                        self.update_single_data((word_list[i-1],best_pair[0]),-freq,pair_freq,pair_words,word_freq,freq_max_heap,word)
                    if i<len(word_list)-1:
                        self.update_single_data((best_pair[1],word_list[i+1]),-freq,pair_freq,pair_words,word_freq,freq_max_heap,word)

                    if i>0:
                        self.update_single_data((word_list[i-1],new_token_bytes),freq,pair_freq,pair_words,word_freq,freq_max_heap,word)
                    if i<len(word_list)-1:
                        self.update_single_data((new_token_bytes,word_list[i+1]),freq,pair_freq,pair_words,word_freq,freq_max_heap,word)
                    

                else:
                    i += 1
            
        if best_pair in pair_freq:
            del pair_freq[best_pair]
        if best_pair in pair_words:
            del pair_words[best_pair]
    
    def update_single_data(self, pairs: Tuple, neg_freq: int, pair_freq: Dict, pair_words: Dict, word_freq: Dict, freq_max_heap: List, word: bytes):
        pair_freq[pairs] += neg_freq

        if pair_freq[pairs] > 0:
            pair_words[pairs].add(word)
            heapq.heappush(freq_max_heap, (-pair_freq[pairs], pairs)) # 注意：这里通常存负频率
        else:
            del pair_freq[pairs]
            if pairs in pair_words:
                pair_words[pairs].discard(word)
                if not pair_words[pairs]:
                    del pair_words[pairs]

# 包装函数保持不变
def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    try:
        with open(input_path, 'r', encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"do not exist file in {input_path}")
        return {}, []

    # 实例化并调用
    tokenizer_instance = BPETokenizer(vocab_size, special_tokens)
    vocab, merged = tokenizer_instance.tokenizer(text) 
    return vocab, merged


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merged: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        self.merged = merged

        self.bytes_to_ids = {v: k for k, v in self.vocab.items()}
        self.merged_ranks = {pair: i for i, pair in enumerate(self.merged)}

        self.special_tokens = {}
        self.special_pattern = None  # ✅ 拼写统一

        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in sorted_tokens:  # ✅ 用 sorted_tokens
                self.special_tokens[token] = self.bytes_to_ids[token.encode("utf-8")]
            self.special_pattern = "(" + "|".join(re.escape(k) for k in sorted_tokens) + ")"

        self.pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    def encode_chunk(self, text_bytes: bytes) -> List[int]:
        pre_tokens = [s.encode("utf-8") for s in self.pat.findall(text_bytes.decode("utf-8", errors="replace"))]
        token_ids: List[int] = []

        for word_bytes in pre_tokens:
            if not word_bytes:
                continue

            parts = [bytes([b]) for b in word_bytes]

            while len(parts) > 1:
                best_merge_info = min(
                    (
                        ((parts[i], parts[i + 1]),
                         self.merged_ranks.get((parts[i], parts[i + 1]), float("inf")))
                        for i in range(len(parts) - 1)
                    ),
                    key=lambda x: x[1],
                )

                if best_merge_info[1] == float("inf"):
                    break

                best_pair_to_merge = best_merge_info[0]

                new_parts = []
                i = 0
                while i < len(parts):  # ✅ 不丢最后一个
                    if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair_to_merge:  # ✅ 用 ==
                        new_parts.append(parts[i] + parts[i + 1])
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts

            # ✅ while 完成后再追加
            for part in parts:
                token_ids.append(self.bytes_to_ids[part])

        return token_ids  # ✅ for 循环结束后再 return

    def encode(self, text: str) -> List[int]:
        if not self.special_pattern:
            return self.encode_chunk(text.encode("utf-8"))

        chunks = re.split(self.special_pattern, text)
        token_ids: List[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                token_ids.append(self.special_tokens[chunk])
            else:
                token_ids.extend(self.encode_chunk(chunk.encode("utf-8")))
        return token_ids

    def decode(self, ids: List[int]) -> str:
        all_bytes = b"".join(self.vocab.get(i, b"") for i in ids)
        return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定字符串可迭代对象，懒惰地生成token ID。
        """
        for text_chunk in iterable:
            encoded_ids = self.encode(text_chunk)
            for token_id in encoded_ids:
                yield token_id

        




