# KiezChat — Claude instructions

## Project structure

- `kiezthropic/` — the Flask chat app (RAG over wiki + Telegram)
- `download_wiki.sh` — downloads wiki from Outline API (reads creds from `.settings`)
- `fetch_telegram.py` — downloads Telegram messages, saves to `kiezthropic/wiki_pages_extra/`
- `telegram_auth.py` — one-time Telethon session auth

## Gitignored data (local only)

- `wiki_pages/` and `kiezthropic/wiki_pages/` — Outline wiki dumps
- `kiezthropic/wiki_pages_extra/` — Telegram exports
- `kiezthropic/faiss_index.bin` + `kiezthropic/chunks.pkl` — built search index
- `.settings` — API credentials
- `kiezthropic/admin_password.txt` — admin dashboard password

Never commit any of these.

## Credentials

Loaded from `.settings` (shell script) or env vars. See README for variable names.
AI Core creds come from `VCAP_SERVICES` on CF or individual `AICORE_*` env vars locally.

## Rebuilding the index

After updating wiki pages or Telegram exports, run:

```bash
cd kiezthropic && python build_index.py
```

## Remotes

- `origin` — SAP GitHub (internal): `git@github.tools.sap:I771869/kiezburn.git`
- `github` — public GitHub: `https://github.com/kryptokommunist/kiezchat.git`

Push to both after changes.
