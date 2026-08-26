from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, math, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ").lower()
    except Exception:
        return text.lower()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c.get("text", "")).split() for c in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception as e:
            print(f"  ⚠️  rank_bm25 unavailable, using internal BM25 fallback: {e}", flush=True)
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=score,
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._memory_docs: list[dict] = []
        self._memory_vectors = None
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
            self.client.get_collections()
        except Exception as e:
            print(f"  ⚠️  Qdrant unavailable, dense search will use memory fallback: {e}", flush=True)

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("USE_LOCAL_DENSE_MODEL") == "1":
                try:
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(EMBEDDING_MODEL)
                except Exception as e:
                    print(f"  ⚠️  Dense encoder unavailable, using lexical fallback: {e}", flush=True)
                    self._encoder = _LexicalEncoder()
            else:
                self._encoder = _LexicalEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        texts = [c.get("text", "") for c in chunks]
        if not texts:
            return

        encoder = self._get_encoder()
        if self.client is None or isinstance(encoder, _LexicalEncoder):
            self._memory_docs = chunks
            self._memory_vectors = encoder.encode(texts, normalize_embeddings=True)
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        try:
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        except Exception:
            try:
                self.client.delete_collection(collection)
            except Exception:
                pass
            self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

        try:
            vectors = encoder.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        except TypeError:
            vectors = self._get_encoder().encode(texts, normalize_embeddings=True)
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**chunks[i].get("metadata", {}), "text": texts[i]},
            )
            for i in range(len(chunks))
        ]
        try:
            self.client.upsert(collection_name=collection, points=points)
        except Exception as e:
            print(f"  ⚠️  Qdrant upsert failed, switching to memory fallback: {e}", flush=True)
            self.client = None
            self._memory_docs = chunks
            self._memory_vectors = vectors

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        encoder = self._get_encoder()
        query_vector = encoder.encode(query, normalize_embeddings=True)
        if hasattr(query_vector, "tolist"):
            query_payload = query_vector.tolist()
        else:
            query_payload = query_vector
        if query_payload and isinstance(query_payload[0], list):
            query_payload = query_payload[0]

        if self.client is None or isinstance(encoder, _LexicalEncoder):
            return self._memory_search(query_payload, top_k)

        try:
            response = self.client.query_points(
                collection_name=collection,
                query=query_payload,
                limit=top_k,
            )
        except Exception as e:
            print(f"  ⚠️  Qdrant query failed, using memory fallback: {e}", flush=True)
            return self._memory_search(query_payload, top_k)
        points = getattr(response, "points", response)
        results = []
        for pt in points:
            payload = pt.payload or {}
            results.append(SearchResult(
                text=payload.get("text", ""),
                score=float(pt.score),
                metadata={k: v for k, v in payload.items() if k != "text"},
                method="dense",
            ))
        return results

    def _memory_search(self, query_vector, top_k: int) -> list[SearchResult]:
        if self._memory_vectors is None:
            return []
        vectors = self._memory_vectors.tolist() if hasattr(self._memory_vectors, "tolist") else self._memory_vectors
        scored = []
        for idx, vector in enumerate(vectors):
            score = _cosine(query_vector, vector)
            scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            doc = self._memory_docs[idx]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=float(score),
                metadata=doc.get("metadata", {}),
                method="dense",
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")


class _SimpleBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / max(len(corpus_tokens), 1)
        self.doc_freq: dict[str, int] = {}
        for doc in corpus_tokens:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        n_docs = len(self.corpus_tokens)
        scores = []
        for doc in self.corpus_tokens:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * len(doc) / max(self.avgdl, 1e-9))
                score += idf * tf * (self.k1 + 1) / denom
            scores.append(score)
        return scores


class _LexicalEncoder:
    def __init__(self):
        self._vocab: dict[str, int] = {}

    def encode(self, texts, normalize_embeddings: bool = True, **_kwargs):
        import numpy as np

        single = isinstance(texts, str)
        if single:
            texts = [texts]
        tokenized = [re.findall(r"\w+", text.lower(), flags=re.UNICODE) for text in texts]
        for tokens in tokenized:
            for token in tokens:
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)

        vectors = np.zeros((len(tokenized), max(len(self._vocab), 1)), dtype=float)
        for row, tokens in enumerate(tokenized):
            for token in tokens:
                vectors[row, self._vocab[token]] += 1.0
            if normalize_embeddings:
                norm = math.sqrt(float(np.dot(vectors[row], vectors[row])))
                if norm:
                    vectors[row] /= norm
        return vectors[0] if single else vectors


def _cosine(a, b) -> float:
    limit = min(len(a), len(b))
    if limit == 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(limit))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(x) * float(x) for x in b))
    return dot / (na * nb) if na and nb else 0.0
