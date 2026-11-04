"""Turn text into vectors.

Two backends behind one small interface:
  - SentenceTransformerEmbedder: real, local, free (default)
  - HashingEmbedder: tiny and offline, for unit tests and CI
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

from swe_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

# BGE models are trained to see this sentence in front of a search QUESTION.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class Embedder(Protocol):
    model_name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Bag-of-words vectors made with a hash. Not smart, but fast and deterministic.

    Texts that share words get similar vectors, which is enough to test our pipeline
    without downloading a model.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self.model_name = f"hashing-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()  # stable across runs
            vec[int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0  # never return an all-zero vector
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Real embeddings on your own CPU. The model loads on first use (lazy)."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.model_name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._max_seq_length = settings.embedding_max_seq_length
        self._model = None

    def _load(self):
        if self._model is None:
            # Imported here so the rest of the app starts fast and tests need no torch.
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._model.max_seq_length = self._max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None)
        return int(getter() if getter else model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # length 1, so cosine = dot product
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self._encode([prefix + text])[0]


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings)