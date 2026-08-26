from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re, math
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD, EMBEDDING_MODEL)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


_MODEL = None


def _get_embedding_model():
    global _MODEL
    if _MODEL is None:
        if os.getenv("USE_LOCAL_SEMANTIC_MODEL") == "1":
            try:
                from sentence_transformers import SentenceTransformer
                _MODEL = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as e:
                print(f"  ⚠️  Embedding model unavailable, using lexical fallback: {e}", flush=True)
                _MODEL = _LexicalEmbeddingModel()
        else:
            _MODEL = _LexicalEmbeddingModel()
    return _MODEL


class _LexicalEmbeddingModel:
    """Tiny deterministic fallback so chunking works without local model deps."""

    def encode(self, texts, normalize_embeddings: bool = True, **_kwargs):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        vocab: dict[str, int] = {}
        tokenized = []
        for text in texts:
            tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            tokenized.append(tokens)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        vectors = np.zeros((len(tokenized), max(len(vocab), 1)), dtype=float)
        for row, tokens in enumerate(tokenized):
            for token in tokens:
                vectors[row, vocab[token]] += 1.0
            if normalize_embeddings:
                norm = math.sqrt(float(np.dot(vectors[row], vectors[row])))
                if norm:
                    vectors[row] /= norm
        return vectors


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?。])\s+|\n{2,}", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_by_size(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks near sentence boundaries, falling back to hard cuts."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), chunk_size):
                piece = sentence[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())
    return chunks


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    import numpy as np

    embeddings = _get_embedding_model().encode(sentences, normalize_embeddings=True)
    groups: list[list[str]] = [[sentences[0]]]

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    chunks = [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": i},
        )
        for i, group in enumerate(groups)
        if " ".join(group).strip()
    ]
    merged: list[Chunk] = []
    for chunk in chunks:
        is_short_header = chunk.text.lstrip().startswith("#") and len(chunk.text) < 80
        if is_short_header and merged:
            merged[-1].text = f"{merged[-1].text} {chunk.text}".strip()
        else:
            merged.append(chunk)
    for i, chunk in enumerate(merged):
        chunk.metadata["chunk_index"] = i
    return merged


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    parent_texts: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > parent_size:
            if current.strip():
                parent_texts.append(current.strip())
                current = ""
            parent_texts.extend(_split_by_size(para, parent_size))
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > parent_size and current:
            parent_texts.append(current.strip())
            current = para
        else:
            current = candidate
    if current.strip():
        parent_texts.append(current.strip())

    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = metadata.get("source", "doc")
    safe_source = re.sub(r"[^A-Za-z0-9_-]+", "_", str(source)).strip("_") or "doc"

    for p_idx, parent_text in enumerate(parent_texts):
        pid = f"{safe_source}_parent_{p_idx}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_idx},
        ))

        child_texts = _split_by_size(parent_text, child_size)
        for c_idx, child_text in enumerate(child_texts):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "parent_id": pid,
                    "child_index": c_idx,
                    "parent_index": p_idx,
                },
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def flush():
        nonlocal current_content
        body = current_content.strip()
        if not body and not current_header:
            return
        full_text = f"{current_header}\n\n{body}".strip() if current_header else body
        if full_text:
            chunks.append(Chunk(
                text=full_text,
                metadata={
                    **metadata,
                    "section": current_header.lstrip("#").strip() if current_header else "root",
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            ))
        current_content = ""

    for part in parts:
        if not part:
            continue
        if re.match(r"^#{1,3}\s+.+$", part.strip()):
            flush()
            current_header = part.strip()
        else:
            current_content += part
    flush()

    if not chunks and text.strip():
        return [
            Chunk(text=c.text, metadata={**c.metadata, "strategy": "structure", "section": "root"})
            for c in chunk_basic(text, metadata=metadata)
        ]
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
