"""Pre-build FAISS index and save to disk. Run this locally before deploying."""
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import faiss
import numpy as np
from fastembed import TextEmbedding

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Wiki page chunking: target word count and sentence overlap
CHUNK_TARGET_WORDS = 400
CHUNK_OVERLAP_SENTENCES = 2

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


def chunk_text(text: str, title: str, source: str) -> list[dict]:
    """Sentence-aware chunking: split on sentence boundaries, accumulate to
    CHUNK_TARGET_WORDS, carry over the last CHUNK_OVERLAP_SENTENCES sentences."""
    raw = re.split(r"(?<=[.!?\n])\s+|(?<=\n)", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        return []

    chunks = []
    current: list[str] = []
    current_words = 0
    seq = 0

    for sent in sentences:
        wc = len(sent.split())
        if current_words + wc > CHUNK_TARGET_WORDS and current:
            chunks.append({"text": " ".join(current), "title": title, "source": source, "seq": seq})
            seq += 1
            current = current[-CHUNK_OVERLAP_SENTENCES:]
            current_words = sum(len(s.split()) for s in current)
        current.append(sent)
        current_words += wc

    if current:
        chunks.append({"text": " ".join(current), "title": title, "source": source, "seq": seq})

    return chunks


def chunk_telegram(text: str, title: str, source: str) -> list[dict]:
    """Split a Telegram export on message boundaries ([YYYY-MM-DD] stamps).

    Each message stays together as one chunk (unless it exceeds CHUNK_TARGET_WORDS,
    in which case it is split normally).  Messages before TELEGRAM_CUTOFF_DATE are
    dropped.  Messages whose heading names BurnHalla are skipped entirely.
    Messages that mention BurnHalla but not Kiez Burn get a [BurnHalla] title tag.
    """
    header_end = TELEGRAM_MSG_RE.search(text)
    body = text[header_end.start():] if header_end else text

    raw_messages = TELEGRAM_MSG_RE.split(body)
    raw_messages = [m.strip() for m in raw_messages if m.strip()]

    date_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]")
    chunks = []
    for msg in raw_messages:
        m = date_re.match(msg)
        if m and m.group(1) < TELEGRAM_CUTOFF_DATE:
            continue
        if BURNHALLA_MSG_RE.match(msg):
            continue

        msg = preprocess(msg)
        words = msg.split()
        if not words:
            continue

        is_burnhalla = bool(BURNHALLA_RE.search(msg))
        is_kiezburn = bool(KIEZBURN_RE.search(msg))
        msg_title = f"{title} [BurnHalla]" if (is_burnhalla and not is_kiezburn) else title

        if len(words) <= CHUNK_TARGET_WORDS:
            chunks.append({"text": msg, "title": msg_title, "source": source})
        else:
            for i in range(0, len(words), CHUNK_TARGET_WORDS - CHUNK_OVERLAP_SENTENCES * 15):
                chunks.append({
                    "text": " ".join(words[i: i + CHUNK_TARGET_WORDS]),
                    "title": msg_title,
                    "source": source,
                })
    return chunks


WIKI_URL_RE = re.compile(r"<!--\s*wiki_url:\s*(https?://\S+)\s*-->")


def build_and_save():
    chunks = []
    for wiki_dir in ["wiki_pages", "wiki_pages_extra"]:
        p = Path(wiki_dir)
        if not p.exists():
            continue
        for md_file in sorted(p.glob("*.md")):
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            title = re.sub(r"_[a-f0-9]{8}$", "", md_file.stem).replace("_", " ")

            url_match = WIKI_URL_RE.search(content)
            wiki_url = url_match.group(1) if url_match else None

            if md_file.name.startswith("telegram_"):
                file_chunks = chunk_telegram(content, title, md_file.name)
            else:
                content = preprocess(content)
                file_chunks = chunk_text(content, title, md_file.name)

            if wiki_url:
                for c in file_chunks:
                    c["url"] = wiki_url

            chunks.extend(file_chunks)

    print(f"Total chunks: {len(chunks)}")

    model = TextEmbedding(EMBED_MODEL, cache_dir="fastembed_cache")
    texts = [c["text"] for c in chunks]
    embeddings = np.array(list(model.embed(texts)), dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors, dim={dim}, model={EMBED_MODEL}")

    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Saved faiss_index.bin and chunks.pkl")


if __name__ == "__main__":
    build_and_save()
