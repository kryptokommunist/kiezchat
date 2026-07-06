"""Pre-build FAISS index and save to disk. Run this locally before deploying."""
import json
import os
import pickle
import re
import sys
from pathlib import Path

# Run from the kiezchat directory
sys.path.insert(0, str(Path(__file__).parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Telegram files have messages separated by [YYYY-MM-DD] date stamps.
# We split on message boundaries so that messages from different events
# (e.g. BurnHalla vs Kiez Burn) never end up in the same chunk.
TELEGRAM_MSG_RE = re.compile(r"(?=\[\d{4}-\d{2}-\d{2}\])")
# Keep only Telegram messages from this date onwards.
# Removes pre-season noise (2025 messages, pre-BurnHalla BurnNight etc.)
# but keeps late-Jan 2026 messages that contain useful Kiez Burn policy info.
TELEGRAM_CUTOFF_DATE = "2026-01-26"
# Messages whose heading starts with "BurnHalla" are about BurnHalla only — skip them.
BURNHALLA_MSG_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s*(?:[🔥✨🎟️❗️⚠️]\s*)?\*?\*?BurnHalla", re.IGNORECASE)
BURNHALLA_RE = re.compile(r"burnhalla", re.IGNORECASE)
KIEZBURN_RE = re.compile(r"kiez.?burn", re.IGNORECASE)


def preprocess(text: str) -> str:
    # Replace wiki.kiezburn.org markdown links with just the link text
    text = re.sub(r'\[([^\]]+)\]\(https?://wiki\.kiezburn\.org/[^\)]+\)', r'\1', text)
    # Remove bare wiki.kiezburn.org URLs
    text = re.sub(r'https?://wiki\.kiezburn\.org/\S+', '', text)
    # Remove mention:// internal CMS links — replace [text](mention://...) with just text
    text = re.sub(r'\[([^\]]+)\]\(mention://[^\)]+\)', r'\1', text)
    # Remove /api/attachments image links entirely (not useful as text)
    text = re.sub(r'\[[^\]]*\]\(/api/attachments[^\)]*\)', '', text)
    text = re.sub(r'/api/attachments\S*', '', text)
    # Remove /doc/ relative links — replace [text](/doc/...) with just text
    text = re.sub(r'\[([^\]]+)\]\(/doc/[^\)]+\)', r'\1', text)
    # Clean up empty markdown link parens left over
    text = re.sub(r'\[([^\]]+)\]\(\s*\)', r'\1', text)
    return text


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


def chunk_telegram(text: str, title: str, source: str) -> list[dict]:
    """Split a Telegram export on message boundaries ([YYYY-MM-DD] stamps).

    Each message stays together as one chunk (unless it exceeds CHUNK_SIZE words,
    in which case it is split normally).  Messages before TELEGRAM_CUTOFF_DATE are
    dropped (they contain BurnHalla and other pre-season noise).  Messages that
    mention BurnHalla but not Kiez Burn are tagged so the LLM can de-prioritise them.
    """
    # Strip the file header (lines before the first date stamp)
    header_end = TELEGRAM_MSG_RE.search(text)
    body = text[header_end.start():] if header_end else text

    # Split into individual messages
    raw_messages = TELEGRAM_MSG_RE.split(body)
    raw_messages = [m.strip() for m in raw_messages if m.strip()]

    date_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]")
    chunks = []
    for msg in raw_messages:
        # Filter out messages before cutoff date
        m = date_re.match(msg)
        if m and m.group(1) < TELEGRAM_CUTOFF_DATE:
            continue

        # Skip messages that are primarily announcements for BurnHalla
        # (these have BurnHalla in the heading after the date stamp)
        if BURNHALLA_MSG_RE.match(msg):
            continue

        msg = preprocess(msg)
        words = msg.split()
        if not words:
            continue

        # Determine event context for this message
        is_burnhalla = bool(BURNHALLA_RE.search(msg))
        is_kiezburn = bool(KIEZBURN_RE.search(msg))
        if is_burnhalla and not is_kiezburn:
            msg_title = f"{title} [BurnHalla]"
        else:
            msg_title = title

        if len(words) <= CHUNK_SIZE:
            chunks.append({"text": msg, "title": msg_title, "source": source})
        else:
            # Long message: split with overlap but keep the title tag
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append({
                    "text": " ".join(words[i: i + CHUNK_SIZE]),
                    "title": msg_title,
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
            title = re.sub(r"_[a-f0-9]{8}$", "", md_file.stem).replace("_", " ")

            # Use message-boundary chunking for Telegram exports
            if md_file.name.startswith("telegram_"):
                file_chunks = chunk_telegram(content, title, md_file.name)
            else:
                content = preprocess(content)
                file_chunks = chunk_text(content, title, md_file.name)

            chunks.extend(file_chunks)

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
