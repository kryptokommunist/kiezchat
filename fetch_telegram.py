"""
Download messages from Berlin Burners and Kiez Burn News Telegram chats.
Anonymizes by stripping sender info. Saves as markdown files for RAG indexing.

Usage:
  python3 fetch_telegram.py

Requires: pip install telethon
Session file must exist at ../telegram-autocalendar/session/telegram_session
"""
from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

API_ID = 32917418
API_HASH = "7dba9a8e2f84e9bf6ee7f47ea2c6993d"
SESSION_PATH = str(Path(__file__).parent / "telegram-autocalendar" / "session" / "telegram_session")
OUT_DIR = Path(__file__).parent / "kiezthropic" / "wiki_pages_extra"

# Chat names to search for (case-insensitive substring match)
CHATS = [
    {"pattern": "berlin burner", "months_back": 3, "slug": "berlin_burners"},
    {"pattern": "kiezburn news", "months_back": 12, "slug": "kiezburn_news"},
    {"pattern": "kiez burn news", "months_back": 12, "slug": "kiezburn_news"},
]


def anonymize(text: str) -> str:
    """Remove @mentions and URLs."""
    text = re.sub(r"@\w+", "@[user]", text)
    return text


async def fetch_chat(client, chat_id: int, name: str, months_back: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
    messages = []
    async for msg in client.iter_messages(chat_id, limit=5000):
        if not msg.text:
            continue
        if msg.date < cutoff:
            break
        date_str = msg.date.strftime("%Y-%m-%d")
        text = anonymize(msg.text.strip())
        messages.append(f"[{date_str}] {text}")
    return messages


async def main():
    try:
        from telethon import TelegramClient
        from telethon.tl.types import Channel, Chat
    except ImportError:
        print("ERROR: telethon not installed. Run: pip install telethon")
        sys.exit(1)

    client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
    await client.start()

    if not await client.is_user_authorized():
        print("ERROR: Not authenticated. Run the telegram-autocalendar app first to create a session.")
        await client.disconnect()
        sys.exit(1)

    print("Fetching dialog list...")
    dialogs = []
    async for dialog in client.iter_dialogs():
        dialogs.append(dialog)
    print(f"Found {len(dialogs)} dialogs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for chat_cfg in CHATS:
        pattern = chat_cfg["pattern"].lower()
        months_back = chat_cfg["months_back"]
        slug = chat_cfg["slug"]
        out_path = OUT_DIR / f"telegram_{slug}.md"

        # Find matching dialog
        match = None
        for d in dialogs:
            if pattern in d.name.lower():
                match = d
                break

        if not match:
            print(f"WARNING: No chat found matching '{pattern}'")
            print("Available chats:")
            for d in dialogs[:30]:
                print(f"  - {d.name}")
            continue

        print(f"Fetching '{match.name}' (last {months_back} months)...")
        messages = await fetch_chat(client, match.id, match.name, months_back)
        print(f"  Got {len(messages)} messages")

        if not messages:
            print(f"  Skipping (empty)")
            continue

        content = f"# Telegram: {match.name}\n"
        content += f"Source: Telegram chat (anonymized, no usernames)\n"
        content += f"Period: last {months_back} months as of {datetime.now().strftime('%Y-%m-%d')}\n\n"
        content += "\n\n".join(messages)

        out_path.write_text(content, encoding="utf-8")
        print(f"  Saved to {out_path}")

    await client.disconnect()
    print("\nDone. Re-run build_index.py to include Telegram data in the RAG index.")


if __name__ == "__main__":
    asyncio.run(main())
