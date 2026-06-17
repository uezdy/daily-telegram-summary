# Daily Telegram Summary

Автоматический пайплайн: раз в сутки читает сообщения из телеграм-группы, формирует краткое резюме с помощью [OpenRouter](https://openrouter.ai) (бесплатные модели) и публикует его в канал через бота.

## Как это работает

1. **Telethon** (user API) загружает текстовые сообщения группы за предыдущие календарные сутки.
2. **OpenRouter** анализирует переписку и выделяет важные обсуждения.
3. **Telegram Bot API** отправляет итоговое сообщение в канал.
4. **GitHub Actions** запускает скрипт каждый день в 09:00 Europe/Minsk.

## Переменные окружения

| Переменная | Описание |
|---|---|
| `TELEGRAM_API_ID` | API ID с [my.telegram.org](https://my.telegram.org) |
| `TELEGRAM_API_HASH` | API Hash с [my.telegram.org](https://my.telegram.org) |
| `TELEGRAM_SESSION` | StringSession Telethon (см. ниже) |
| `TELEGRAM_GROUP` | Группа-источник: `@username` или `-100...` |
| `TELEGRAM_BOT_TOKEN` | Токен бота от [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHANNEL` | Канал назначения: `@channel` или `-100...` |
| `OPENROUTER_API_KEY` | API-ключ [OpenRouter](https://openrouter.ai/keys) |
| `TIMEZONE` | Часовой пояс для границ «суток» (по умолчанию `Europe/Minsk`) |

Скопируйте `.env.example` в `.env` для локального запуска.

## Подготовка

### 1. Telegram API и сессия

```bash
pip install -r requirements.txt
cp .env.example .env
# заполните TELEGRAM_API_ID и TELEGRAM_API_HASH в .env
python scripts/create_session.py
```

Скопируйте выведенную строку в `TELEGRAM_SESSION`. Аккаунт должен быть участником группы-источника.

### 2. Бот и канал

1. Создайте бота через [@BotFather](https://t.me/BotFather).
2. Добавьте бота в канал как администратора с правом публикации.
3. Укажите `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHANNEL`.

### 3. OpenRouter API

1. Зарегистрируйтесь на [openrouter.ai](https://openrouter.ai).
2. Создайте ключ на [openrouter.ai/keys](https://openrouter.ai/keys).
3. Укажите `OPENROUTER_API_KEY` в `.env` и GitHub Secrets.

Модель зашита в коде: `openrouter/free` (бесплатный роутер OpenRouter). Список бесплатных моделей: [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0).

### 4. GitHub Secrets

В репозитории: **Settings → Secrets and variables → Actions → New repository secret**

Добавьте все переменные из таблицы выше.

Опционально в **Variables** добавьте `TIMEZONE`.

## Локальный запуск

```bash
pip install -r requirements.txt
python scripts/daily_summary.py
```

## Расписание

По умолчанию workflow запускается **со вторника по субботу в 09:00 Europe/Minsk** (06:00 UTC). Чтобы изменить время, отредактируйте cron в `.github/workflows/daily-summary.yml`:

```yaml
- cron: '0 6 * * 2-6'  # минуты часы день_месяца месяц день_недели (UTC)
```

Примеры:

- `0 6 * * 2-6` — 06:00 UTC (09:00 Minsk), вт–сб
- `0 6 * * *` — 06:00 UTC (09:00 Minsk), каждый день
- `40 12 * * *` — 12:40 UTC (15:40 Minsk), каждый день

Ручной запуск: **Actions → Daily Telegram Summary → Run workflow**.

## Структура

```
scripts/
  telegram_common.py   # env, клиент Telethon, период суток
  create_session.py    # генерация TELEGRAM_SESSION
  daily_summary.py     # основной пайплайн
.github/workflows/
  daily-summary.yml    # ежедневный cron
```
