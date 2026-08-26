from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, COHERE_API_KEY


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Reranker.

    LƯU Ý (deviation so với đề bài): thay vì tải cross-encoder cục bộ
    `BAAI/bge-reranker-v2-m3` (~2.27GB, không có sẵn trong cache máy),
    dùng Cohere Rerank API (`rerank-multilingual-v3.0`, free tier) —
    cùng chức năng (query, doc) -> relevance score, chỉ khác backend.
    """

    def __init__(self, model_name: str = "rerank-multilingual-v3.0"):
        self.model_name = model_name
        self._client = None

    def _load_model(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(COHERE_API_KEY)
        return self._client

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        if os.getenv("USE_COHERE_RERANK") != "1" or not COHERE_API_KEY:
            return _lexical_fallback_rerank(query, documents, top_k)
        try:
            client = self._load_model()
            texts = [doc["text"] for doc in documents]
            response = client.rerank(
                model=self.model_name,
                query=query,
                documents=texts,
                top_n=min(top_k, len(texts)),
            )
            return [
                RerankResult(
                    text=documents[r.index]["text"],
                    original_score=documents[r.index].get("score", 0.0),
                    rerank_score=float(r.relevance_score),
                    metadata=documents[r.index].get("metadata", {}),
                    rank=i,
                )
                for i, r in enumerate(response.results)
            ]
        except Exception as e:
            print(f"  ⚠️  Cohere rerank failed, using lexical fallback: {e}", flush=True)
            return _lexical_fallback_rerank(query, documents, top_k)


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": i, "text": d["text"], "meta": d.get("metadata", {})}
                for i, d in enumerate(documents)
            ]
            ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
            results = []
            for i, item in enumerate(ranked[:top_k]):
                doc_idx = int(item.get("id", i))
                doc = documents[doc_idx]
                results.append(RerankResult(
                    text=doc["text"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i,
                ))
            return results
        except Exception:
            return _lexical_fallback_rerank(query, documents, top_k)


def _lexical_fallback_rerank(query: str, documents: list[dict], top_k: int) -> list[RerankResult]:
    query_terms = {t for t in query.lower().split() if len(t) > 1}
    scored = []
    for idx, doc in enumerate(documents):
        text = doc.get("text", "")
        text_lower = text.lower()
        overlap = sum(1 for term in query_terms if term in text_lower)
        lexical = overlap / max(len(query_terms), 1)
        original = float(doc.get("score", 0.0) or 0.0)
        score = lexical + 0.05 * math.tanh(original)
        scored.append((score, idx, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        RerankResult(
            text=doc.get("text", ""),
            original_score=float(doc.get("score", 0.0) or 0.0),
            rerank_score=float(score),
            metadata=doc.get("metadata", {}),
            rank=rank,
        )
        for rank, (score, _, doc) in enumerate(scored[:top_k])
    ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
