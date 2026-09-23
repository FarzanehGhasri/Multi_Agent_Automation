"""Embedding adapters - concrete implementations of :class:`EmbeddingModel`.

Two interchangeable implementations are provided (Liskov Substitution in
practice: the vector store cannot tell them apart):

* :class:`HashingTfidfEmbeddings` - dependency-free, deterministic, offline.
  It is the default so the notebook runs identically on any machine, with no
  API key, no network access and no model download.
* :class:`LangChainEmbeddingsAdapter` - wraps any LangChain ``Embeddings``
  object (OpenAI, HuggingFace, Cohere, ...) behind the very same port.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Sequence

#: Words carrying no discriminative power in SOP documents (EN + FA).
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been by for from has have in is it its of on or that
    the this to was were will with within must may shall not no every each all
    و در به از با که این آن را برای های ها می شود باید است بر تا یا هر
    """.split()
)

#: Unicode-aware tokenizer: keeps Latin and Persian/Arabic letters and digits.
_TOKEN_RE = re.compile(r"[0-9a-z؀-ۿ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case, split into word tokens and drop stop-words.

    Exposed as a module-level function because both the embedder and the
    keyword fallback of the retriever need exactly the same tokenisation
    (Don't Repeat Yourself, and it keeps the two stages consistent).
    """
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _bucket(token: str, dimensions: int) -> int:
    """Stable hash of a token into a vector slot.

    ``hash()`` is salted per interpreter run, so a cryptographic digest is used
    instead: the same corpus must always produce the same vectors.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


class HashingTfidfEmbeddings:
    """TF-IDF vectors projected into a fixed-size space by the hashing trick.

    Why not a neural embedder? Because for a small, highly structured corpus
    of SOP documents, lexical TF-IDF retrieval is both sufficient and exactly
    reproducible - and it keeps the deliverable runnable offline. The port it
    implements is what matters; swapping in a hosted embedding model is a
    one-line change in the composition root.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._idf: dict[str, float] = {}
        self._default_idf: float = 1.0
        self._fitted = False

    @property
    def name(self) -> str:
        return f"hashing-tfidf-{self._dimensions}d"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def fit(self, texts: Sequence[str]) -> "HashingTfidfEmbeddings":
        """Compute the inverse document frequency table over the corpus."""
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(tokenize(text)))
        total = max(1, len(texts))
        # Smoothed IDF: never zero, never undefined for unseen tokens.
        self._idf = {
            token: math.log((1 + total) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }
        self._default_idf = math.log(1 + total) + 1.0
        self._fitted = True
        return self

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Fit (once) on the corpus and return one vector per document."""
        if not self._fitted:
            self.fit(texts)
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the IDF table learned from the corpus."""
        return self._vectorize(text)

    def _vectorize(self, text: str) -> list[float]:
        """Sub-linear TF x IDF, hashed into buckets, then L2-normalised."""
        counts = Counter(tokenize(text))
        vector = [0.0] * self._dimensions
        for token, count in counts.items():
            weight = (1.0 + math.log(count)) * self._idf.get(token, self._default_idf)
            vector[_bucket(token, self._dimensions)] += weight
        return _l2_normalise(vector)


class LangChainEmbeddingsAdapter:
    """Adapter that makes any LangChain ``Embeddings`` satisfy our port.

    The adapter pattern keeps the vendor type out of the domain and out of the
    agents: only this file knows that LangChain exists.
    """

    def __init__(self, embeddings: Any, name: str = "langchain-embeddings") -> None:
        self._embeddings = embeddings
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self._embeddings.embed_documents(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._embeddings.embed_query(text)))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, safe for zero vectors."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


__all__ = [
    "HashingTfidfEmbeddings",
    "LangChainEmbeddingsAdapter",
    "cosine_similarity",
    "tokenize",
]
