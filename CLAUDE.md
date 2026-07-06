# KiezChat — Claude instructions

## Project structure

- `kiezchat/` — the Flask chat app (RAG over wiki + Telegram)
- `download_wiki.sh` — downloads wiki from Outline API (reads creds from `.settings`)
- `fetch_telegram.py` — downloads Telegram messages, saves to `kiezchat/wiki_pages_extra/`
- `telegram_auth.py` — one-time Telethon session auth

## Gitignored data (local only)

- `wiki_pages/` and `kiezchat/wiki_pages/` — Outline wiki dumps
- `kiezchat/wiki_pages_extra/` — Telegram exports
- `kiezchat/faiss_index.bin` + `kiezchat/chunks.pkl` — built search index
- `.settings` — API credentials
- `kiezchat/admin_password.txt` — admin dashboard password

Never commit any of these.

## Credentials

Loaded from `.settings` (shell script) or env vars. See README for variable names.
AI Core creds come from `VCAP_SERVICES` on CF or individual `AICORE_*` env vars locally.

## Rebuilding the index

After updating wiki pages or Telegram exports, run:

```bash
cd kiezchat && python build_index.py
```

## Remotes

- `github` — public GitHub: `https://github.com/kryptokommunist/kiezchat.git`
