# KiezChat

A RAG-powered chat assistant for [Kiez Burn](https://kiezburn.org) — answers questions about the event using the Outline wiki and Telegram channels as a knowledge base.

The app is deployed on SAP Cloud Foundry and uses SAP AI Core (GPT-4o) for inference.

## Architecture

```
wiki_pages/          ← Outline wiki dumps (local only, gitignored)
kiezchat/
  wiki_pages/        ← wiki pages used by the app (local only, gitignored)
  wiki_pages_extra/  ← Telegram chat exports (local only, gitignored)
  app.py             ← Flask app, agentic RAG loop, AI Core integration
  rag.py             ← BM25 + FAISS hybrid retrieval
  build_index.py     ← preprocesses markdown, chunks text, builds index
  corrections_seed.json ← approved answer corrections, seeded into SQLite on deploy
  templates/         ← chat UI, admin dashboard
download_wiki.sh     ← downloads wiki pages from Outline API
fetch_telegram.py    ← downloads Telegram messages (Berlin Burners, KB News)
telegram_auth.py     ← one-time Telethon session auth helper
```

## How the RAG works

### 1. Data sources

Two sources are combined into a single index:

- **Wiki pages** (`wiki_pages/`) — exported from the Outline wiki at wiki.kiezburn.org via the API. One markdown file per page.
- **Telegram exports** (`wiki_pages_extra/`) — messages from the Berlin Burners and Kiez Burn News channels, fetched with Telethon and saved as anonymized markdown (sender info stripped).

### 2. Preprocessing (`build_index.py`)

Each markdown file is cleaned before chunking:

- Internal wiki links (`[text](https://wiki.kiezburn.org/...)`) are reduced to their link text.
- CMS-internal links (`mention://...`, `/doc/...`) are stripped, keeping only the display text.
- Image attachment links (`/api/attachments/...`) are removed entirely.
- The page title is derived from the filename by removing the 8-character Outline UUID suffix and replacing underscores with spaces.

Text is then split into overlapping word-level chunks (400 words, 50-word overlap) so long pages don't exceed the context window and adjacent chunks share some context.

### 3. Index (`build_index.py`)

All chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors, L2-normalized) and stored in a FAISS `IndexFlatIP` (inner-product / cosine similarity). The index and chunk metadata are saved to `faiss_index.bin` and `chunks.pkl`, both loaded into memory at app startup.

### 4. Retrieval (`rag.py`)

Each query runs two searches in parallel and merges the results:

- **Vector search** — embeds the query with the same MiniLM model, finds the top-K nearest chunks by cosine similarity.
- **BM25 keyword search** — tokenizes the query and scores all chunks using BM25Okapi (rank-bm25).

Chunks that appear in both result sets are ranked first (tagged `"both"`), followed by vector-only hits sorted by cosine score. Up to `top_k * 2` results are returned when both methods are combined.

### 5. Agentic answer loop (`app.py`)

The app uses a two-phase agentic loop rather than a single retrieval-then-answer step:

1. **Search phase** — the model is given a `search()` tool and an `add_to_context()` tool. It calls `search()` with specific terms (up to 3 times), inspects truncated snippets, then calls `add_to_context()` with the IDs of the most relevant chunks to fetch their full text.
2. **Answer phase** — once tool calls are exhausted, the model generates a streaming answer using the full chunk text as context.

The system prompt injects any approved corrections from the SQLite database so known bad answers are overridden regardless of what the retrieved chunks say.

## Setup

### 1. Credentials

Copy `.settings.example` to `.settings` (gitignored) and fill in your values:

```bash
WIKI_API_TOKEN=<your Outline API token>
WIKI_BASE_URL=https://wiki.kiezburn.org/api
WIKI_COLLECTION_ID_2026=<collection UUID>
```

For local development, also export AI Core credentials:

```bash
export AICORE_CLIENT_ID=...
export AICORE_CLIENT_SECRET=...
export AICORE_AUTH_URL=...
export AICORE_API_URL=...
```

### 2. Obtain wiki pages

```bash
./download_wiki.sh
```

This downloads all wiki pages from Outline into `wiki_pages/`. Copy or symlink them into `kiezchat/wiki_pages/`.

### 3. Obtain Telegram chat logs (optional)

Authenticate once (creates a session file):

```bash
python3 telegram_auth.py
```

Then fetch messages:

```bash
python3 fetch_telegram.py
```

Output goes to `kiezchat/wiki_pages_extra/` as anonymized markdown files.

### 4. Build the search index

```bash
cd kiezchat
pip install -r requirements.txt
python build_index.py
```

This produces `faiss_index.bin` and `chunks.pkl` (both gitignored).

### 5. Run locally

```bash
cd kiezchat
flask run
# or
gunicorn app:app
```

Set `ADMIN_PASSWORD` env var or create `kiezchat/admin_password.txt` (gitignored) for the admin dashboard at `/admin`.

## Deployment (SAP Cloud Foundry)

```bash
cd kiezchat
cf push
```

The app reads credentials from `VCAP_SERVICES` automatically when running on CF.
