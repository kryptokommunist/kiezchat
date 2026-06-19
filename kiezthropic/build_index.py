"""Pre-build FAISS index and save to disk. Run this locally before deploying."""
import json
import os
import pickle
import sys
from pathlib import Path

# Run from the kiezthropic directory
sys.path.insert(0, str(Path(__file__).parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def chunk_text(text, title, source):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append({
            "text": " ".join(words[i: i + CHUNK_SIZE]),
            "title": title,
            "source": source,
        })
    return chunks

def build_and_save():
    chunks = []
    for wiki_dir in ["wiki_pages", "wiki_pages_extra"]:
        p = Path(wiki_dir)
        if not p.exists():
            continue
        for md_file in sorted(p.glob("*.md")):
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            import re
            title = re.sub(r"_[a-f0-9]{8}$", "", md_file.stem).replace("_", " ")
            chunks.extend(chunk_text(content, title, md_file.name))

    print(f"Total chunks: {len(chunks)}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Saved faiss_index.bin and chunks.pkl")

if __name__ == "__main__":
    build_and_save()
