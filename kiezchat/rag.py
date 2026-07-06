"""RAG retriever using pre-built FAISS index + fastembed (ONNX, no PyTorch)."""
from __future__ import annotations

import os
import pickle
import re
import threading
from pathlib import Path

import faiss
import numpy as np

TOP_K = 5

# Chunks whose source filename contains a year other than EVENT_YEAR are excluded
# at load time. Set EVENT_YEAR="" to disable filtering.
EVENT_YEAR: str = os.environ.get("EVENT_YEAR", "2026")

_index: faiss.IndexFlatIP | None = None
_chunks: list[dict] = []
_embed_fn = None
_base_dir: str = ""
_bm25 = None
_bm25_corpus: list[list[str]] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from fastembed import TextEmbedding
        cache_dir = str(Path(_base_dir) / "fastembed_cache")
        model = TextEmbedding(
            "BAAI/bge-small-en-v1.5",
            cache_dir=cache_dir,
        )
        def _fn(texts: list[str]) -> np.ndarray:
            embs = list(model.query_embed(texts))
            arr = np.array(embs, dtype="float32")
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.maximum(norms, 1e-9)
        _embed_fn = _fn
    return _embed_fn


def _get_bm25():
    global _bm25, _bm25_corpus
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        _bm25_corpus = [_tokenize(c["title"] + " " + c["text"]) for c in _chunks]
        _bm25 = BM25Okapi(_bm25_corpus)
    return _bm25


_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                try:
                    from sentence_transformers.cross_encoder import CrossEncoder
                    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2", max_length=512)
                    print("Cross-encoder reranker loaded.")
                except Exception as e:
                    print(f"Reranker unavailable: {e}")
                    _reranker = False
    return _reranker if _reranker is not False else None


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Re-score candidates with a cross-encoder; falls back to original order."""
    reranker = _get_reranker()
    if not reranker or not candidates:
        return candidates
    try:
        pairs = [(query, c["text"][:512]) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        for score, chunk in ranked:
            chunk["rerank_score"] = float(score)
        return [chunk for _, chunk in ranked]
    except Exception as e:
        print(f"Reranking failed: {e}")
        return candidates


def _year_in_filename(filename: str) -> str | None:
    """Return a 4-digit year string if one appears in the filename, else None."""
    m = re.search(r"(20\d{2})", filename)
    return m.group(1) if m else None


def load_prebuilt(base_dir: str) -> None:
    global _index, _chunks, _base_dir
    _base_dir = base_dir
    index_path = Path(base_dir) / "faiss_index.bin"
    chunks_path = Path(base_dir) / "chunks.pkl"
    full_index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        all_chunks: list[dict] = pickle.load(f)

    if EVENT_YEAR:
        kept_faiss_ids = []
        kept_chunks = []
        for faiss_idx, chunk in enumerate(all_chunks):
            year = _year_in_filename(chunk.get("source", ""))
            if year is None or year == EVENT_YEAR:
                kept_chunks.append(chunk)
                kept_faiss_ids.append(faiss_idx)
        skipped = len(all_chunks) - len(kept_chunks)
        if skipped:
            print(f"Year filter (EVENT_YEAR={EVENT_YEAR}): skipped {skipped} chunks from other years")
        _chunks = kept_chunks

        # Build a sub-index with only the kept vectors so chunk IDs stay sequential
        dim = full_index.d
        sub_index = faiss.IndexFlatIP(dim)
        all_vecs = faiss.rev_swig_ptr(full_index.get_xb(), full_index.ntotal * dim)
        all_vecs = np.frombuffer(all_vecs, dtype="float32").reshape(full_index.ntotal, dim)
        sub_index.add(all_vecs[kept_faiss_ids])
        _index = sub_index
    else:
        _chunks = all_chunks
        _index = full_index

    _build_source_ranges()
    print(f"Loaded pre-built index: {_index.ntotal} vectors, {len(_chunks)} chunks (EVENT_YEAR={EVENT_YEAR or 'all'})")


# ---------------------------------------------------------------------------
# Source-range map — built at load time, used for context expansion
# ---------------------------------------------------------------------------

_source_ranges: dict[str, tuple[int, int]] = {}


def _build_source_ranges() -> None:
    global _source_ranges
    _source_ranges = {}
    current_src: str | None = None
    start = 0
    for i, c in enumerate(_chunks):
        src = c["source"]
        if src != current_src:
            if current_src is not None:
                _source_ranges[current_src] = (start, i - 1)
            current_src = src
            start = i
    if current_src is not None:
        _source_ranges[current_src] = (start, len(_chunks) - 1)


def expand_context(ids: list[int], window: int = 1) -> list[int]:
    """Return ids expanded to include ±window neighbors within the same source."""
    expanded: set[int] = set(ids)
    for idx in ids:
        if idx < 0 or idx >= len(_chunks):
            continue
        src = _chunks[idx]["source"]
        lo, hi = _source_ranges.get(src, (idx, idx))
        for offset in range(-window, window + 1):
            neighbor = idx + offset
            if lo <= neighbor <= hi:
                expanded.add(neighbor)
    return sorted(expanded)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Vector search only."""
    if _index is None or not _chunks:
        return []
    embed = _get_embed_fn()
    q_emb = embed([query])
    scores, indices = _index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = _chunks[idx].copy()
        chunk["score"] = float(score)
        chunk["idx"] = int(idx)
        chunk["match"] = "vector"
        results.append(chunk)
    return results


def retrieve_bm25(query: str, top_k: int = TOP_K) -> list[dict]:
    """BM25 keyword search."""
    if not _chunks:
        return []
    bm25 = _get_bm25()
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunk = _chunks[idx].copy()
        chunk["score"] = float(scores[idx])
        chunk["idx"] = int(idx)
        chunk["match"] = "keyword"
        results.append(chunk)
    return results


RRF_K = 60
MAX_CHUNKS_PER_SOURCE = 2


def _source_dedup(results: list[dict], max_per_source: int = MAX_CHUNKS_PER_SOURCE) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for chunk in results:
        src = chunk.get("source", "")
        if seen.get(src, 0) < max_per_source:
            out.append(chunk)
            seen[src] = seen.get(src, 0) + 1
    return out


def retrieve_combined(query: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid vector + BM25 search merged with Reciprocal Rank Fusion."""
    vec_results = retrieve(query, top_k=top_k)
    bm25_results = retrieve_bm25(query, top_k=top_k)

    rrf_scores: dict[int, float] = {}
    for rank, c in enumerate(vec_results):
        rrf_scores[c["idx"]] = rrf_scores.get(c["idx"], 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, c in enumerate(bm25_results):
        rrf_scores[c["idx"]] = rrf_scores.get(c["idx"], 0.0) + 1.0 / (RRF_K + rank + 1)

    all_candidates: dict[int, dict] = {}
    for c in vec_results + bm25_results:
        if c["idx"] not in all_candidates:
            all_candidates[c["idx"]] = c

    vec_set = {c["idx"] for c in vec_results}
    bm25_set = {c["idx"] for c in bm25_results}

    sorted_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[: top_k * 2]
    results = []
    for idx in sorted_ids:
        chunk = all_candidates[idx].copy()
        chunk["rrf_score"] = rrf_scores[idx]
        chunk["match"] = (
            "both" if (idx in vec_set and idx in bm25_set)
            else ("vector" if idx in vec_set else "keyword")
        )
        results.append(chunk)

    return _source_dedup(results)


def get_chunks_by_ids(ids: list[int]) -> list[dict]:
    """Return full chunks for the given FAISS index positions."""
    result = []
    for i in ids:
        if 0 <= i < len(_chunks):
            chunk = _chunks[i].copy()
            chunk["idx"] = i
            result.append(chunk)
    return result


def retrieve_by_source(source_substring: str) -> list[dict]:
    """Return all chunks whose source filename contains the given substring."""
    return [
        dict(c, idx=i)
        for i, c in enumerate(_chunks)
        if source_substring in c.get("source", "")
    ]
