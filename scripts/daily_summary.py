#!/usr/bin/env python3
"""Fetch group messages for the last day, summarize with AI, post to a channel."""

from __future__ import annotations

import asyncio
import html
import json
import os
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
YANDEXGPT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEXGPT_DEFAULT_MODEL = "yandexgpt-lite"
MAX_YANDEXGPT_RETRIES = 5
YANDEXGPT_RETRYABLE_STATUS_CODES = {429, 503}
TELEGRAM_ALLOWED_HTML_TAGS = frozenset({"b", "i", "a", "strong", "em"})
TELEGRAM_HTML_TAG_PATTERN = r"</?(?:a|b|i)(?:\s[^>]*)?>"
MSG_MARKER_PATTERN = re.compile(r"\[\[msg:(\d+)\]\]|\[msg:(\d+)\](?!\])")
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


def get_yandexgpt_model() -> str:
    model = os.environ.get("YANDEXGPT_MODEL", YANDEXGPT_DEFAULT_MODEL).strip()
    return model or YANDEXGPT_DEFAULT_MODEL


def request_yandexgpt_completion(
    api_key: str,
    folder_id: str,
    model: str,
    prompt: str,
) -> dict:
    model_uri = f"gpt://{folder_id}/{model}"
    for attempt in range(1, MAX_YANDEXGPT_RETRIES + 1):
        response = httpx.post(
            YANDEXGPT_API_URL,
            headers={
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "modelUri": model_uri,
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.3,
                    "maxTokens": 2048,
                },
                "messages": [{"role": "user", "text": prompt}],
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
            response.status_code in YANDEXGPT_RETRYABLE_STATUS_CODES
            and attempt < MAX_YANDEXGPT_RETRIES
        ):
            wait_seconds = parse_retry_after_seconds(response)
            print(
                f"YandexGPT rate limited ({model}), "
                f"retrying in {wait_seconds:.0f}s "
                f"({attempt}/{MAX_YANDEXGPT_RETRIES})..."
            )
            time.sleep(wait_seconds)
            log_openrouter_request(model, prompt, attempt + 1)
            continue

        detail = response.text.strip()
        raise RuntimeError(
            f"YandexGPT request failed ({response.status_code}): {detail}"
        )

    raise RuntimeError(f"YandexGPT request failed after {MAX_YANDEXGPT_RETRIES} retries")


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
- После важного тезиса добавляй ссылку на источник маркером [[msg:ID]] (1–3 ID на тезис). Ставь маркер после текста тезиса, вне HTML-тегов (не внутри <b> или <i>). Используй только ID из входных [msg:ID]. Не выдумывай ID.
- Только теги <b> и <i>. Без <ul>, <ol>, <li>, <p>, <br>, <a>, без Markdown (** или __).
- Без markdown code blocks (```). Начинай сразу с HTML, без обёртки.
- Только переносы строк между блоками.

Пример:
<b>{SUMMARY_HEADER}</b>

<b>{DEFAULT_GENERAL_TOPIC_TITLE}</b>
• <b>Поиск людей</b>: Сложность из-за законов о данных. [[msg:123]]

<b>Библиотека/Ссылки</b>
• <b>Новые материалы</b>: Опубликовали подборку по архивам. [[msg:456]]

Сообщения:
{messages_text}
"""


def strip_markdown_code_fences(text: str) -> str:
    """Remove ``` wrappers that some LLMs add around the response."""
    normalized = text.strip()
    normalized = re.sub(r"^```(?:\w+)?\s*\n?", "", normalized)
    normalized = re.sub(r"\n?```\s*$", "", normalized)
    return normalized.strip()


def normalize_telegram_html(text: str) -> str:
    """Convert common Markdown patterns to Telegram HTML."""
    normalized = strip_markdown_code_fences(text)
    normalized = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", normalized)
    normalized = re.sub(r"__(.+?)__", r"<b>\1</b>", normalized)
    normalized = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", normalized)
    normalized = re.sub(r"<br\s*/?>", "\n", normalized, flags=re.IGNORECASE)
    return sanitize_telegram_html(normalized)


def escape_telegram_html_text(text: str) -> str:
    """Escape raw <, >, & outside allowed Telegram HTML tags."""
    parts = re.split(f"({TELEGRAM_HTML_TAG_PATTERN})", text, flags=re.IGNORECASE)
    escaped: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            escaped.append(part)
            continue
        escaped.append(
            re.sub(r"&(?!amp;|lt;|gt;|quot;|#\d+;)", "&amp;", part)
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    return "".join(escaped)


def sanitize_telegram_html(text: str) -> str:
    """Keep only Telegram-supported tags and fix common entity issues."""
    sanitized = re.sub(r"<strong>", "<b>", text, flags=re.IGNORECASE)
    sanitized = re.sub(r"</strong>", "</b>", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<em>", "<i>", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"</em>", "</i>", sanitized, flags=re.IGNORECASE)
    sanitized = escape_telegram_html_text(sanitized)

    allowed = "|".join(sorted(TELEGRAM_ALLOWED_HTML_TAGS))
    sanitized = re.sub(
        rf"<(?!/)(?!{allowed}\b)[^>]*>",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        rf"</(?!{allowed}\b)[^>]*>",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def strip_html_tags(text: str) -> str:
    plain = strip_markdown_code_fences(text)
    plain = re.sub(r"<[^>]+>", "", plain)
    return html.unescape(plain)


def html_to_plain_with_links(text: str) -> str:
    def link_replacer(match: re.Match[str]) -> str:
        url = html.unescape(match.group(1))
        return f" ↗ {url}"

    with_links = re.sub(
        r'<a href="([^"]+)">↗</a>',
        link_replacer,
        text,
        flags=re.IGNORECASE,
    )
    return strip_html_tags(with_links)


def move_links_outside_formatting(text: str) -> str:
    """Telegram HTML does not allow <a> tags inside <b> or <i>."""
    for _ in range(5):
        updated = re.sub(
            r'<([bi])>((?:(?!\1>).)*?)<a href="([^"]+)">↗</a>((?:(?!\1>).)*?)</\1>',
            r'<a href="\3">↗</a> <\1>\2\4</\1>',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if updated == text:
            return text
        text = updated
    return text


def inject_message_links(text: str, message_links: dict[int, str]) -> str:
    def replace_marker(match: re.Match[str]) -> str:
        raw_id = match.group(1) or match.group(2)
        message_id = int(raw_id)
        url = message_links.get(message_id)
        if not url:
            return ""
        escaped_url = html.escape(url, quote=True)
        return f'<a href="{escaped_url}">↗</a>'

    text = MSG_MARKER_PATTERN.sub(replace_marker, text)
    return move_links_outside_formatting(text)


def summarize_with_yandexgpt(
    messages_text: str,
    period_label: str,
    message_links: dict[int, str],
) -> str:
    api_key = require_env("YANDEX_CLOUD_API_KEY")
    folder_id = require_env("YANDEX_CLOUD_FOLDER_ID")
    model = get_yandexgpt_model()
    prompt = build_summary_prompt(messages_text, period_label)
    payload = request_yandexgpt_completion(api_key, folder_id, model, prompt)

    try:
        summary = payload["result"]["alternatives"][0]["message"]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected YandexGPT response: {payload}") from exc

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

    attempts: list[tuple[str, str | None]] = [
        (sanitize_telegram_html(text), "HTML"),
        (html_to_plain_with_links(text), None),
    ]

    last_error = "unknown error"
    for index, (message_text, parse_mode) in enumerate(attempts):
        if index > 0:
            print("Telegram rejected HTML formatting, retrying as plain text...")

        payload: dict[str, object] = {
            "chat_id": channel,
            "text": message_text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

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
    model = get_yandexgpt_model()
    print(f"Generating summary with YandexGPT ({model})...")
    summary = summarize_with_yandexgpt(messages_text, period_label, message_links)
    return ensure_summary_header(summary)


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
