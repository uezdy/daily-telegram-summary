#!/usr/bin/env python3
"""Fetch group messages for the last day, summarize with AI, post to a channel."""

from __future__ import annotations

import asyncio
import html
import json
import re
import sys
import time
from collections import defaultdict

import httpx

from telegram_common import (
    DEFAULT_GENERAL_TOPIC_TITLE,
    FetchedMessage,
    GENERAL_TOPIC_ID,
    GroupFetchResult,
    build_message_link,
    fetch_group_messages_for_period,
    get_summary_period,
    get_timezone,
    require_env,
)

MAX_TELEGRAM_MESSAGE_LENGTH = 4096
MAX_SUMMARY_LENGTH = 3800
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/free"
MAX_OPENROUTER_RETRIES = 5
MAX_OPENROUTER_EMPTY_CONTENT_RETRIES = 3
OPENROUTER_RETRYABLE_STATUS_CODES = {429, 503}
SUMMARY_HEADER = "Уезды Беларуси, обсуждения за сутки:"
SUMMARY_HEADER_HTML = f"<b>{SUMMARY_HEADER}</b>"
LOG_PREVIEW_MAX_CHARS = 4000


def print_section(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n {title}\n{line}")


def print_text_block(label: str, text: str, max_chars: int = LOG_PREVIEW_MAX_CHARS) -> None:
    print(f"{label} ({len(text)} chars):")
    if len(text) <= max_chars:
        print(text)
        return
    print(text[:max_chars])
    print(f"... [truncated, {len(text) - max_chars} more chars]")


def print_json_block(label: str, data: object, max_chars: int = LOG_PREVIEW_MAX_CHARS) -> None:
    formatted = json.dumps(data, ensure_ascii=False, indent=2)
    print_text_block(label, formatted, max_chars=max_chars)


def build_message_links(
    messages: list[FetchedMessage],
    username: str | None,
    chat_id: int,
) -> dict[int, str]:
    return {
        message.id: build_message_link(
            username,
            chat_id,
            message.topic_id,
            message.id,
        )
        for message in messages
    }


def build_messages_text(
    messages: list[FetchedMessage],
    topic_titles: dict[int, str],
    tz,
) -> str:
    by_topic: dict[int, list[FetchedMessage]] = defaultdict(list)
    for message in messages:
        by_topic[message.topic_id].append(message)

    topic_ids = sorted(
        by_topic.keys(),
        key=lambda topic_id: (topic_id != GENERAL_TOPIC_ID, topic_id),
    )

    sections: list[str] = []
    for topic_id in topic_ids:
        title = topic_titles.get(topic_id, f"Тема {topic_id}")
        lines = [f"=== {title} (topic_id={topic_id}) ==="]
        for message in by_topic[topic_id]:
            local_time = message.date.astimezone(tz).strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"[msg:{message.id}] [{local_time}] {message.sender}: {message.text}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def parse_retry_after_seconds(response: httpx.Response) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass

    try:
        metadata = response.json().get("error", {}).get("metadata", {})
        retry_after_seconds = metadata.get("retry_after_seconds")
        if retry_after_seconds is not None:
            return max(float(retry_after_seconds), 1.0)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass

    return 20.0


def log_openrouter_request(model: str, prompt: str, attempt: int) -> None:
    print_section(f"OpenRouter request (attempt {attempt}/{MAX_OPENROUTER_RETRIES})")
    print(f"URL: {OPENROUTER_API_URL}")
    print(f"Model: {model}")
    print(f"Temperature: 0.3, max_tokens: 2048")
    print_text_block("Prompt", prompt)


def log_openrouter_response(payload: dict) -> None:
    print_section("OpenRouter API response")
    print(f"Response id: {payload.get('id')}")
    print(f"Model: {payload.get('model')}")

    usage = payload.get("usage")
    if usage:
        print_json_block("Usage", usage, max_chars=500)

    choices = payload.get("choices") or []
    print(f"Choices: {len(choices)}")
    for index, choice in enumerate(choices):
        message = choice.get("message") or {}
        content = message.get("content")
        print(f"Choice {index}: finish_reason={choice.get('finish_reason')!r}")
        if choice.get("error"):
            print_json_block(f"Choice {index} error", choice["error"], max_chars=2000)
        if content is not None:
            print_text_block(f"Choice {index} content", str(content))


def request_openrouter_completion(
    api_key: str,
    model: str,
    prompt: str,
    *,
    attempt: int = 1,
) -> dict:
    log_openrouter_request(model, prompt, attempt)
    for attempt in range(attempt, MAX_OPENROUTER_RETRIES + 1):
        response = httpx.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/uezdy/daily-telegram-summary",
                "X-Title": "Daily Telegram Summary",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=120.0,
        )
        if response.is_success:
            payload = response.json()
            log_openrouter_response(payload)
            return payload

        print_section(f"OpenRouter API error ({response.status_code})")
        print_text_block("Response body", response.text.strip(), max_chars=8000)

        if (
            response.status_code in OPENROUTER_RETRYABLE_STATUS_CODES
            and attempt < MAX_OPENROUTER_RETRIES
        ):
            wait_seconds = parse_retry_after_seconds(response)
            print(
                f"OpenRouter rate limited ({model}), "
                f"retrying in {wait_seconds:.0f}s "
                f"({attempt}/{MAX_OPENROUTER_RETRIES})..."
            )
            time.sleep(wait_seconds)
            log_openrouter_request(model, prompt, attempt + 1)
            continue

        detail = response.text.strip()
        raise RuntimeError(
            f"OpenRouter request failed ({response.status_code}): {detail}"
        )

    raise RuntimeError(f"OpenRouter request failed after {MAX_OPENROUTER_RETRIES} retries")


def extract_openrouter_content(payload: dict) -> str:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {payload}") from exc

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if text:
                    parts.append(text)
        content = "\n".join(parts)

    if not isinstance(content, str):
        finish_reason = choice.get("finish_reason")
        error = choice.get("error")
        raise RuntimeError(
            "OpenRouter returned empty content "
            f"(finish_reason={finish_reason!r}, error={error!r})"
        )

    summary = content.strip()
    if not summary:
        finish_reason = choice.get("finish_reason")
        raise RuntimeError(
            f"OpenRouter returned blank content (finish_reason={finish_reason!r})"
        )
    return summary


def build_summary_prompt(messages_text: str, period_label: str) -> str:
    return f"""Ты помощник для телеграм-сообщества. Проанализируй сообщения из группы за сутки ({period_label}).

Входные данные сгруппированы по темам форума (topic). Каждое сообщение помечено как [msg:ID].

Задача:
1. Выдели важные обсуждения, решения, объявления и открытые вопросы.
2. Игнорируй флуд, мемы и мелкий оффтоп, если они не важны для сообщества.
3. Сформируй одно связное резюме для публикации в телеграм-канале.
4. Пиши на русском языке, кратко и по делу.
5. Не превышай {MAX_SUMMARY_LENGTH} символов.

Формат (строго HTML Telegram, parse_mode=HTML):
- Заголовок: <b>{SUMMARY_HEADER}</b>
- Группируй резюме по темам форума из входных данных. Каждая тема — отдельный блок с заголовком <b>Название темы</b> (без topic_id).
- Пропускай темы без значимых обсуждений за сутки.
- Внутри темы: тезисы через «• <b>Краткий заголовок</b>: описание»; подпункты через «- ».
- После важного тезиса добавляй ссылку на источник маркером [[msg:ID]] (1–3 ID на тезис). Используй только ID из входных [msg:ID]. Не выдумывай ID.
- Если есть полезные рекомендации — общий блок «Советы:» со списком «- » (можно со ссылками [[msg:ID]]).
- В конце: Игнорируем: флуд, мемы, мелкий оффтоп.
- Только теги <b> и <i>. Без <ul>, <ol>, <li>, <p>, <br>, <a>, без Markdown (** или __).
- Только переносы строк между блоками.

Пример:
<b>{SUMMARY_HEADER}</b>

<b>{DEFAULT_GENERAL_TOPIC_TITLE}</b>
• <b>Поиск людей</b>: Сложность из-за законов о данных. [[msg:123]]

<b>Библиотека/Ссылки</b>
• <b>Новые материалы</b>: Опубликовали подборку по архивам. [[msg:456]]

Советы:
- Проверяйте правила отделения почты. [[msg:789]]

Игнорируем: флуд, мемы, мелкий оффтоп.

Сообщения:
{messages_text}
"""


def normalize_telegram_html(text: str) -> str:
    """Convert common Markdown patterns to Telegram HTML."""
    normalized = text.strip()
    normalized = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", normalized)
    normalized = re.sub(r"__(.+?)__", r"<b>\1</b>", normalized)
    normalized = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", normalized)
    normalized = re.sub(r"<br\s*/?>", "\n", normalized, flags=re.IGNORECASE)
    return normalized


def inject_message_links(text: str, message_links: dict[int, str]) -> str:
    def replace_marker(match: re.Match[str]) -> str:
        message_id = int(match.group(1))
        url = message_links.get(message_id)
        if not url:
            return ""
        escaped_url = html.escape(url, quote=True)
        return f'<a href="{escaped_url}">↗</a>'

    return re.sub(r"\[\[msg:(\d+)\]\]", replace_marker, text)


def summarize_with_openrouter(
    messages_text: str,
    period_label: str,
    message_links: dict[int, str],
) -> str:
    api_key = require_env("OPENROUTER_API_KEY")
    prompt = build_summary_prompt(messages_text, period_label)

    summary = ""
    for attempt in range(1, MAX_OPENROUTER_EMPTY_CONTENT_RETRIES + 1):
        payload = request_openrouter_completion(api_key, OPENROUTER_MODEL, prompt)
        try:
            summary = extract_openrouter_content(payload)
            print_section("OpenRouter extracted summary (raw)")
            print_text_block("Summary", summary)
            break
        except RuntimeError as exc:
            if attempt >= MAX_OPENROUTER_EMPTY_CONTENT_RETRIES:
                raise
            print(
                f"{exc}, retrying "
                f"({attempt}/{MAX_OPENROUTER_EMPTY_CONTENT_RETRIES})..."
            )
            time.sleep(2.0)

    summary = normalize_telegram_html(summary)
    print_section("Summary after HTML normalization")
    print_text_block("Summary", summary)

    summary = inject_message_links(summary, message_links)
    print_section("Summary after message link injection")
    print_text_block("Summary", summary)

    if len(summary) > MAX_TELEGRAM_MESSAGE_LENGTH:
        print(
            f"Summary exceeds Telegram limit "
            f"({len(summary)} > {MAX_TELEGRAM_MESSAGE_LENGTH}), truncating..."
        )
        summary = summary[: MAX_TELEGRAM_MESSAGE_LENGTH - 3] + "..."

    return summary


def ensure_summary_header(summary: str) -> str:
    summary = re.sub(
        r"<b>\s*(?:Важные обсуждения за сутки|Уезды Беларуси, обсуждения за сутки)"
        r"[^<]*</b>\s*\n*",
        "",
        summary,
        count=1,
        flags=re.IGNORECASE,
    ).lstrip("\n")
    if SUMMARY_HEADER not in summary:
        summary = f"{SUMMARY_HEADER_HTML}\n\n{summary}"
    return summary


def build_empty_summary() -> str:
    return (
        f"{SUMMARY_HEADER_HTML}\n\n"
        "За последние сутки в группе не было текстовых сообщений."
    )


def send_to_channel(text: str) -> None:
    bot_token = require_env("TELEGRAM_BOT_TOKEN")
    channel = require_env("TELEGRAM_CHANNEL")
    send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload: dict[str, object] = {
        "chat_id": channel,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    print_section("Telegram message to send")
    print(f"Channel: {channel}")
    print(f"Parse mode: HTML")
    print(f"Length: {len(text)} chars (limit {MAX_TELEGRAM_MESSAGE_LENGTH})")
    print_text_block("Message text", text, max_chars=MAX_TELEGRAM_MESSAGE_LENGTH)

    response = httpx.post(send_url, json=payload, timeout=60.0)

    print_section("Telegram API response")
    print(f"Status: {response.status_code}")
    try:
        print_json_block("Response body", response.json(), max_chars=8000)
    except json.JSONDecodeError:
        print_text_block("Response body", response.text.strip(), max_chars=8000)

    if not response.is_success:
        raise RuntimeError(
            f"Failed to send message to channel: "
            f"{response.status_code} {response.text.strip()}"
        )


def build_summary(
    fetch_result: GroupFetchResult,
    period_label: str,
    tz,
) -> str:
    messages = fetch_result.messages
    if not messages:
        print("No messages to summarize, using empty summary template.")
        return build_empty_summary()

    messages_text = build_messages_text(messages, fetch_result.topic_titles, tz)
    if not messages_text.strip():
        print("Messages text is empty after filtering, using empty summary template.")
        return build_empty_summary()

    print_section("Input for OpenRouter")
    print(f"Messages: {len(messages)}")
    print_json_block("Topic titles", fetch_result.topic_titles, max_chars=2000)
    print_text_block("Messages text", messages_text)

    message_links = build_message_links(
        messages,
        fetch_result.username,
        fetch_result.chat_id,
    )
    print_section("Message links for summary")
    print_json_block("Links", message_links, max_chars=8000)

    print(f"Generating summary with OpenRouter ({OPENROUTER_MODEL})...")
    summary = summarize_with_openrouter(messages_text, period_label, message_links)
    summary = ensure_summary_header(summary)

    print_section("Final summary (ready for Telegram)")
    print_text_block("Summary", summary, max_chars=MAX_TELEGRAM_MESSAGE_LENGTH)

    return summary


async def run() -> None:
    group = require_env("TELEGRAM_GROUP")
    tz = get_timezone()
    period_start, period_end = get_summary_period(tz)

    period_label = (
        f"{period_start.astimezone(tz).strftime('%d.%m.%Y')} — "
        f"{period_end.astimezone(tz).strftime('%d.%m.%Y')}"
    )

    print_section("Daily summary run")
    print(f"Group: {group}")
    print(f"Timezone: {tz}")
    print(f"Period (local): {period_label}")
    print(
        f"Period (UTC): {period_start.isoformat()} — {period_end.isoformat()}"
    )

    print(f"\nFetching messages for period: {period_label}")
    fetch_result = await fetch_group_messages_for_period(
        group,
        period_start,
        period_end,
    )

    topic_ids = {message.topic_id for message in fetch_result.messages}
    print_section("Fetched messages")
    print(f"Chat id: {fetch_result.chat_id}")
    print(f"Username: {fetch_result.username or '(private)'}")
    print(f"Messages: {len(fetch_result.messages)}")
    print(f"Topics with messages: {len(topic_ids)}")
    for topic_id in sorted(topic_ids, key=lambda tid: (tid != GENERAL_TOPIC_ID, tid)):
        title = fetch_result.topic_titles.get(topic_id, f"Тема {topic_id}")
        count = sum(1 for message in fetch_result.messages if message.topic_id == topic_id)
        print(f"  - {title} (topic_id={topic_id}): {count} messages")

    summary = build_summary(fetch_result, period_label, tz)

    print("\nSending summary to channel...")
    send_to_channel(summary)
    print_section("Done")
    print("Summary posted successfully.")


def main() -> None:
    try:
        asyncio.run(run())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
