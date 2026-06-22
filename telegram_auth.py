"""
Interactive Telegram authentication — creates the session file.
Run this ONCE interactively, then fetch_telegram.py will work without auth.

Usage: python3 telegram_auth.py
"""
import asyncio
import sys
from pathlib import Path

API_ID = 32917418
API_HASH = "7dba9a8e2f84e9bf6ee7f47ea2c6993d"
SESSION_DIR = Path(__file__).parent / "telegram-autocalendar" / "session"
SESSION_PATH = str(SESSION_DIR / "telegram_session")


async def main():
    try:
        from telethon import TelegramClient
    except ImportError:
        print("Install telethon first: pip install telethon")
        sys.exit(1)

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(SESSION_PATH, API_ID, API_HASH)

    print("Connecting to Telegram...")
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Already logged in as: {me.first_name} ({me.phone})")
        await client.disconnect()
        return

    phone = input("Enter your phone number (e.g. +49123456789): ").strip()
    await client.send_code_request(phone)
    code = input("Enter the code you received: ").strip()

    try:
        await client.sign_in(phone, code)
    except Exception as e:
        if "2FA" in str(e) or "password" in str(e).lower():
            pwd = input("Enter your 2FA password: ").strip()
            await client.sign_in(password=pwd)
        else:
            raise

    me = await client.get_me()
    print(f"\nLogged in as: {me.first_name} ({me.phone})")
    print(f"Session saved to: {SESSION_PATH}.session")
    await client.disconnect()
    print("\nNow run: python3 fetch_telegram.py")


if __name__ == "__main__":
    asyncio.run(main())
