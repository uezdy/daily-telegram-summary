"""Shared Telegram configuration and helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from telethon import TelegramClient, utils
from telethon.sessions import StringSession
from telethon.tl.functions.messages import GetForumTopicsRequest
from telethon.tl.types import ForumTopic, Message

GENERAL_TOPIC_ID = 1
DEFAULT_GENERAL_TOPIC_TITLE = "Обсуждения"


@dataclass(frozen=True)
class FetchedMessage:
    id: int
    date: datetime
    sender: str
    text: str
    topic_id: int


@dataclass(frozen=True)
class GroupFetchResult:
    messages: list[FetchedMessage]
    topic_titles: dict[int, str]
    username: str | None
    chat_id: int


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_timezone() -> ZoneInfo:
    tz_name = os.environ.get("TIMEZONE", "Europe/Minsk").strip() or "Europe/Minsk"
    try:
        return ZoneInfo(tz_name)
    except Exception as exc:
        hint = ""
        if os.name == "nt":
            hint = " On Windows install IANA timezones: pip install tzdata"
        raise RuntimeError(f"Invalid TIMEZONE: {tz_name}.{hint}") from exc


def get_summary_period(tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Return UTC bounds for the previous calendar day in the given timezone."""
    now_local = datetime.now(tz)
    period_end_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    period_start_local = period_end_local - timedelta(days=1)

    period_start = period_start_local.astimezone(timezone.utc)
    period_end = period_end_local.astimezone(timezone.utc)
    return period_start, period_end


def create_user_client() -> TelegramClient:
    api_id = int(require_env("TELEGRAM_API_ID"))
    api_hash = require_env("TELEGRAM_API_HASH")
    session = require_env("TELEGRAM_SESSION")
    return TelegramClient(StringSession(session), api_id, api_hash)


def format_sender_name(sender) -> str:
    if sender is None:
        return "unknown"

    username = getattr(sender, "username", None)
    if username:
        return f"@{username}"

    first_name = getattr(sender, "first_name", None) or ""
    last_name = getattr(sender, "last_name", None) or ""
    full_name = f"{first_name} {last_name}".strip()
    return full_name or "unknown"


def get_message_topic_id(message: Message) -> int:
    reply = message.reply_to
    if reply is not None:
        top_id = getattr(reply, "reply_to_top_id", None)
        if top_id:
            return top_id
        if getattr(reply, "forum_topic", False):
            reply_to_msg_id = getattr(reply, "reply_to_msg_id", None)
            if reply_to_msg_id:
                return reply_to_msg_id
    return GENERAL_TOPIC_ID


def internal_chat_id(chat_id: int) -> int:
    normalized = abs(chat_id)
    prefix = str(normalized)
    if prefix.startswith("100"):
        return int(prefix[3:])
    return normalized


def build_message_link(
    username: str | None,
    chat_id: int,
    topic_id: int,
    message_id: int,
) -> str:
    if username:
        if topic_id != GENERAL_TOPIC_ID:
            return f"https://t.me/{username}/{topic_id}/{message_id}"
        return f"https://t.me/{username}/{message_id}"

    internal_id = internal_chat_id(chat_id)
    if topic_id != GENERAL_TOPIC_ID:
        return f"https://t.me/c/{internal_id}/{topic_id}/{message_id}"
    return f"https://t.me/c/{internal_id}/{message_id}"


async def fetch_topic_titles(client: TelegramClient, entity) -> dict[int, str]:
    titles = {GENERAL_TOPIC_ID: DEFAULT_GENERAL_TOPIC_TITLE}
    try:
        result = await client(
            GetForumTopicsRequest(
                peer=entity,
                q="",
                offset_date=None,
                offset_id=0,
                offset_topic=0,
                limit=100,
            )
        )
        for topic in result.topics:
            if isinstance(topic, ForumTopic):
                titles[topic.id] = topic.title
    except Exception:
        pass
    return titles


async def fetch_group_messages_for_period(
    group: str,
    period_start: datetime,
    period_end: datetime,
) -> GroupFetchResult:
    client = create_user_client()
    messages: list[FetchedMessage] = []

    async with client:
        entity = await client.get_entity(group)
        topic_titles = await fetch_topic_titles(client, entity)
        username = getattr(entity, "username", None) or None
        chat_id = utils.get_peer_id(entity)

        async for message in client.iter_messages(entity, offset_date=period_end):
            if not isinstance(message, Message):
                continue
            if message.date < period_start:
                break
            if message.date >= period_end:
                continue

            text = (message.message or "").strip()
            if not text:
                continue

            messages.append(
                FetchedMessage(
                    id=message.id,
                    date=message.date,
                    sender=format_sender_name(message.sender),
                    text=text,
                    topic_id=get_message_topic_id(message),
                )
            )

    messages.reverse()
    return GroupFetchResult(
        messages=messages,
        topic_titles=topic_titles,
        username=username,
        chat_id=chat_id,
    )
