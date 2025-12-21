from optparse import Values
import regex as re
from collections import Counter,defaultdict
from typing import List,Dict,Tuple,Set

from torch import special

class Pretokenizer:
    def __init__(self,special_tokens: List[str]):
        self.special_tokens = {token:token.encode('utf-8') for token in special_tokens}
        self.word_pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens.keys(),key=len,reverse=True)
            special_pat_str = "|".join(re.escape(token) for token in sorted_special_tokens)
            self.special_pat = re.compile(f'({special_pat_str})')
        else:
            self.special_pat = None

    def tokenize(self,text: str)-> List[bytes]:

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

    def __init__(self,vocab_size:int,special_tokens:str):
        if vocab_size < 256:
            raise ValueError("vocab_size is smaller than 256")

        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.pretokenizer = Pretokenizer(special_tokens)

        self.vocab:Dict[int,bytes] = {i:bytes(i) for i in range(256)}
        for token in self.special_tokens:
            encode_token = token.encode('utf-8')
            if encode_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = encode_token

    def tokenizer(self,text:str)->Tuple[Dict[int,bytes],List[Tuple[bytes,bytes]]]:
        all_tokens = self.pretokenizer.tokenize(text)
        special_tokens = set(self.pretokenizer.special_tokens.values())
        word_freq = Counter(token for token in all_tokens if token not in special_tokens)

        splits:Dict[bytes,List[bytes]] = {token:[bytes([b]) for b in token] for token in word_freq}
        pair_freq:Dict[Tuple[bytes,bytes],int] = defaultdict(int)
        pair_words:Dict[Tuple[bytes,bytes],Set(bytes)] = defaultdict(set)
        freq_max_heap:List[int,Tuple[bytes,bytes]] = []

        for word,freq in word_freq:
            word_split = splits[word]
            if len(word_split)>1:
                for p1,p2 in zip(word_split[:-1],word_split[1:]):
                    pair = (p1,p2)
                    pair_freq[pair] += freq
                    pair_words[pair].add(word)
        
        for pair,freq in pair_freq:
            heap.heappush(freq_max_heap,(-freq,pair))
        
        merged:List[Tuple[bytes,bytes]] = []
        num_merged_pair = self.vocab_size-len(self.vocab)
        while len(merged)<num_merged_pair:
            best_pair = self.find_best_pair(freq_max_heap,pair_freq)
            if best_pair is None:
                break

            merged.append(best_pair)
            new_token_bytes = best_pair[0]+best_pair[1]
            self.vocab[len(self.vocab)] = new_token_bytes

            self.update_data_struct(best_pair, new_token_bytes, splits, pair_freq, pair_words, word_freq, freq_max_heap)

        
        return self.vocab,merged

        

        










