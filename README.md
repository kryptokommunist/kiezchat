# KiezChat

A RAG-powered chat assistant for [Kiez Burn](https://kiezburn.org) — answers questions about the event using the Outline wiki and Telegram channels as a knowledge base.

The app is deployed on SAP Cloud Foundry and uses SAP AI Core (GPT-4o) for inference.

## Architecture

```
wiki_pages/          ← Outline wiki dumps (local only, gitignored)
kiezchat/
  wiki_pages/        ← symlinked / copied wiki pages for the app (local only)
  wiki_pages_extra/  ← Telegram chat exports (local only)
  app.py             ← Flask app, RAG pipeline, AI Core integration
  rag.py             ← BM25 + FAISS hybrid retrieval
  build_index.py     ← builds faiss_index.bin + chunks.pkl from wiki pages
  templates/         ← chat UI, admin dashboard
download_wiki.sh     ← downloads wiki pages from Outline API
fetch_telegram.py    ← downloads Telegram messages (Berlin Burners, KB News)
telegram_auth.py     ← one-time Telegram session auth helper
```

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
