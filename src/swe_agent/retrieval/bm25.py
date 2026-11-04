"""BM25 keyword search, written by hand so every number is explainable."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from swe_agent.retrieval.tokenizer import tokenize_code
from swe_agent.schemas.code import CodeChunk
from swe_agent.schemas.retrieval import SearchHit


class Bm25Index:
    def __init__(self, chunks: list[CodeChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._term_counts: list[Counter[str]] = []
        self._lengths: list[int] = []
        doc_freq: Counter[str] = Counter()
        self._postings: dict[str, list[int]] = defaultdict(list)  # word -> chunk numbers

        for number, chunk in enumerate(self._chunks):
            tokens = tokenize_code(chunk.embed_text)  # includes the file/name header
            counts = Counter(tokens)
            self._term_counts.append(counts)
            self._lengths.append(len(tokens))
            doc_freq.update(counts.keys())
            for term in counts:
                self._postings[term].append(number)

        total = len(self._chunks)
        self._avg_len = sum(self._lengths) / total if total else 0.0
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        terms = set(tokenize_code(query))
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue  # the word appears nowhere in the repo
            for number in self._postings[term]:
                tf = self._term_counts[number][term]
                length_ratio = self._lengths[number] / self._avg_len
                norm = tf + self._k1 * (1 - self._b + self._b * length_ratio)
                scores[number] += idf * tf * (self._k1 + 1) / norm

        # Sort by score (high first). Ties are broken by position, so results are repeatable.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            SearchHit(chunk=self._chunks[number], score=score, source="bm25")
            for number, score in ranked
        ]