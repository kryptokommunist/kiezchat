"""RAG retriever using pre-built FAISS index + fastembed (ONNX, no PyTorch)."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
import numpy as np

TOP_K = 5

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
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir,
        )
        def _fn(texts: list[str]) -> np.ndarray:
            embs = list(model.embed(texts))
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


def load_prebuilt(base_dir: str) -> None:
    global _index, _chunks, _base_dir
    _base_dir = base_dir
    index_path = Path(base_dir) / "faiss_index.bin"
    chunks_path = Path(base_dir) / "chunks.pkl"
    _index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)
    print(f"Loaded pre-built index: {_index.ntotal} vectors, {len(_chunks)} chunks")


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


def retrieve_combined(query: str, top_k: int = TOP_K) -> list[dict]:
    """Run vector + BM25 in parallel and merge by idx, deduplicating."""
    vec_results = retrieve(query, top_k=top_k)
    bm25_results = retrieve_bm25(query, top_k=top_k)

    seen: dict[int, dict] = {}
    for c in vec_results:
        seen[c["idx"]] = c

    for c in bm25_results:
        idx = c["idx"]
        if idx in seen:
            seen[idx]["match"] = "both"
        else:
            seen[idx] = c

    # Sort: "both" first, then by vector score desc (bm25 scores aren't comparable)
    combined = list(seen.values())
    combined.sort(key=lambda x: (x["match"] != "both", -x.get("score", 0)))
    return combined[:top_k * 2]  # allow more results when combining


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
