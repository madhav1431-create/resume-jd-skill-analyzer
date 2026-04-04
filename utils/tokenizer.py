"""
tokenizer.py
------------
Simple word-level tokenizer built from scratch (no external NLP libraries).

Capabilities:
  • Build vocabulary from a corpus
  • Encode text → token-id sequence (with padding / truncation)
  • Decode token-ids → text
  • Save / load vocabulary to JSON

Special tokens:
  <PAD>  = 0   padding
  <UNK>  = 1   unknown word
  <BOS>  = 2   beginning of sequence (optional)
  <EOS>  = 3   end of sequence (optional)
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Union

import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """
    Lowercase, keep alphanumeric + common punctuation, collapse whitespace.
    """
    text = text.lower()
    # Keep letters, digits, +, #, ., /  (useful for skill names like C++, C#)
    text = re.sub(r"[^a-z0-9#+./\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens (words)."""
    return _clean(text).split()


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOKENIZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class WordTokenizer:
    """
    Word-level tokenizer that builds its vocabulary from a corpus.

    Example
    -------
    >>> tok = WordTokenizer(max_vocab=8000)
    >>> tok.build(corpus_texts)
    >>> ids = tok.encode("Python and machine learning", max_len=32)
    >>> tok.decode(ids)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, max_vocab: int = 10_000):
        self.max_vocab = max_vocab
        self._word2id: dict[str, int] = {}
        self._id2word: dict[int, str] = {}
        self._built = False

    # ── Build / Fit ────────────────────────────────────────────────────────

    def build(self, texts: list[str], min_freq: int = 2) -> "WordTokenizer":
        """
        Fit vocabulary on a list of raw strings.

        Parameters
        ----------
        texts    : list of raw text strings
        min_freq : minimum token frequency to be included in vocab
        """
        counter: Counter = Counter()
        for text in texts:
            counter.update(_tokenize(text))

        # Initialise with special tokens
        self._word2id = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}

        # Add most-frequent words up to max_vocab
        for word, freq in counter.most_common(self.max_vocab - len(self.SPECIAL_TOKENS)):
            if freq < min_freq:
                break
            if word not in self._word2id:
                self._word2id[word] = len(self._word2id)

        self._id2word = {i: w for w, i in self._word2id.items()}
        self._built = True
        return self

    # ── Encode / Decode ────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        max_len: int = 256,
        add_special: bool = True,
    ) -> list[int]:
        """
        Convert text → list of token ids, padded / truncated to `max_len`.

        add_special: prepend BOS and append EOS before padding/truncating.
        """
        assert self._built, "Call build() first."
        tokens = _tokenize(text)

        if add_special:
            # Reserve 2 slots for BOS + EOS
            tokens = tokens[: max_len - 2]
            ids = [self.BOS_ID] + [self._word2id.get(t, self.UNK_ID) for t in tokens] + [self.EOS_ID]
        else:
            tokens = tokens[:max_len]
            ids = [self._word2id.get(t, self.UNK_ID) for t in tokens]

        # Pad to max_len
        pad_len = max_len - len(ids)
        ids = ids + [self.PAD_ID] * pad_len
        return ids

    def encode_batch(
        self,
        texts: list[str],
        max_len: int = 256,
    ) -> torch.Tensor:
        """
        Encode a list of texts → LongTensor of shape (B, max_len).
        """
        ids = [self.encode(t, max_len) for t in texts]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token ids back to a space-joined string."""
        special_ids = {self.PAD_ID, self.BOS_ID, self.EOS_ID}
        words = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            words.append(self._id2word.get(i, self.UNK_TOKEN))
        return " ".join(words)

    def make_padding_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean mask  True = PAD position (to be ignored by attention).
        Shape: (B, seq_len)
        """
        return token_ids == self.PAD_ID

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"max_vocab": self.max_vocab, "word2id": self._word2id}, f, indent=2)
        print(f"[Tokenizer] Saved vocab ({len(self._word2id)} tokens) → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WordTokenizer":
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(max_vocab=data["max_vocab"])
        tok._word2id = {w: int(i) for w, i in data["word2id"].items()}
        tok._id2word = {int(i): w for w, i in tok._word2id.items()}
        tok._built = True
        print(f"[Tokenizer] Loaded vocab ({len(tok._word2id)} tokens) from {path}")
        return tok

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._word2id)

    def __repr__(self) -> str:
        status = f"vocab_size={self.vocab_size}" if self._built else "not built"
        return f"WordTokenizer({status})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corpus = [
        "Experienced data scientist skilled in python machine learning and pytorch.",
        "Looking for a backend engineer with python django and postgresql experience.",
        "NLP engineer with deep learning transformers and huggingface expertise.",
    ]
    tok = WordTokenizer(max_vocab=500)
    tok.build(corpus, min_freq=1)
    print(tok)

    ids = tok.encode("python machine learning", max_len=16)
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))

    batch = tok.encode_batch(corpus[:2], max_len=20)
    print("Batch shape:", batch.shape)
    print("Padding mask:\n", tok.make_padding_mask(batch))
