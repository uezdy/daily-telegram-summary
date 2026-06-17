#!/usr/bin/env python3
"""Create a Telethon StringSession for TELEGRAM_SESSION env variable."""

from __future__ import annotations

import asyncio
import os
import sys

from telethon import TelegramClient
from telethon.sessions import StringSession

from telegram_common import require_env


async def main() -> None:
    api_id = int(require_env("TELEGRAM_API_ID"))
    api_hash = require_env("TELEGRAM_API_HASH")

    client = TelegramClient(StringSession(), api_id, api_hash)
    await client.start()

    session_string = client.session.save()
    await client.disconnect()

    print("\nAdd this value to GitHub Secrets and local .env as TELEGRAM_SESSION:\n")
    print(session_string)
    print(
        "\nKeep this string secret. Anyone with it can access your Telegram account."
    )


if __name__ == "__main__":
    if not os.environ.get("TELEGRAM_API_ID") or not os.environ.get("TELEGRAM_API_HASH"):
        print(
            "Set TELEGRAM_API_ID and TELEGRAM_API_HASH before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(main())
