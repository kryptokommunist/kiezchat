"""RAG retriever using pre-built FAISS index + fastembed (ONNX, no PyTorch)."""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

TOP_K = 5

_index: faiss.IndexFlatIP | None = None
_chunks: list[dict] = []
_embed_fn = None
_base_dir: str = ""


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
        results.append(chunk)
    return results


def retrieve_by_source(source_substring: str) -> list[dict]:
    """Return all chunks whose source filename contains the given substring."""
    return [c.copy() for c in _chunks if source_substring in c.get("source", "")]
