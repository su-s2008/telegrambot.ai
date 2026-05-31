import asyncio
import io
import json
import logging
import os
import platform
import requests
import socket
import sys
import time
from collections import defaultdict
from typing import Any

from PIL import Image
from telebot.async_telebot import AsyncTeleBot
from telebot.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

try:
    from google import genai as google_genai
    from google.genai import types as google_types

    HAS_GOOGLE_GENAI = True
except ModuleNotFoundError:
    google_types = None
    HAS_GOOGLE_GENAI = False

legacy_genai = None
HAS_LEGACY_GEMINI = False
if not HAS_GOOGLE_GENAI:
    try:
        import google.generativeai as legacy_genai

        HAS_LEGACY_GEMINI = True
    except ModuleNotFoundError:
        pass

try:
    import g4f

    HAS_G4F = True
except ModuleNotFoundError:
    HAS_G4F = False

try:
    import pytesseract

    HAS_TESSERACT = True
except ModuleNotFoundError:
    HAS_TESSERACT = False


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram-ai-bot")


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using %s.", name, raw_value, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s is too small. Using %s.", name, value, default)
        return default
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid float for %s=%r. Using %s.", name, raw_value, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s is too small. Using %s.", name, value, default)
        return default
    return value


BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
ADMIN_USER_ID = env_int("ADMIN_USER_ID", 0, minimum=0)

MAX_CONTEXT_MESSAGES = env_int("MAX_CONTEXT_MESSAGES", 14, minimum=2)
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 8000, minimum=500)
MAX_USER_CHARS = env_int("MAX_USER_CHARS", 12000, minimum=500)
MAX_IMAGE_SIDE = env_int("MAX_IMAGE_SIDE", 1280, minimum=256)
AI_TIMEOUT_SECONDS = env_int("AI_TIMEOUT_SECONDS", 70, minimum=5)
COOLDOWN_SECONDS = env_float("COOLDOWN_SECONDS", 0.8, minimum=0)
TELEGRAM_LIMIT = 4096
AI_CONCURRENCY = env_int("AI_CONCURRENCY", 3, minimum=1)
ENABLE_G4F_FALLBACK = os.getenv("ENABLE_G4F_FALLBACK", "0").lower() in {"1", "true", "yes", "on"}
SINGLE_INSTANCE_PORT = env_int("SINGLE_INSTANCE_PORT", 49383, minimum=1024)
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "Ти корисний Telegram-асистент. Відповідай українською, живо, "
        "дружньо і по суті. Не вигадуй фактів. Якщо не впевнений, чесно "
        "скажи про це. Для коду давай короткі пояснення і робочі приклади."
    ),
)

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN не знайдено. Додайте його у змінні середовища або файл .env."
    )

bot = AsyncTeleBot(BOT_TOKEN)

chat_contexts: dict[int, list[dict[str, str]]] = defaultdict(list)
chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
ai_semaphore = asyncio.Semaphore(AI_CONCURRENCY)
last_request_time: dict[int, float] = defaultdict(float)
last_user_message: dict[int, str] = {}
translate_pending: dict[int, str] = {}
image_analyses: dict[int, dict[str, str | None]] = {}
feedback_store: dict[tuple[int, int], str] = {}
single_instance_socket: socket.socket | None = None
known_users: dict[str, dict[str, Any]] = {}


def load_known_users() -> None:
    global known_users
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as users_file:
            data = json.load(users_file)
    except (OSError, json.JSONDecodeError):
        known_users = {}
        return

    if isinstance(data, dict):
        known_users = data
    else:
        known_users = {}


def save_known_users() -> None:
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as users_file:
            json.dump(known_users, users_file, ensure_ascii=False, indent=2)
    except OSError:
        logger.exception("Could not save users file")


def track_user(message_or_call: Any) -> None:
    user = getattr(message_or_call, "from_user", None)
    if user is None and getattr(message_or_call, "message", None):
        user = getattr(message_or_call.message, "from_user", None)
    if user is None:
        return

    user_id = str(user.id)
    known_users[user_id] = {
        "id": user.id,
        "username": getattr(user, "username", None),
        "first_name": getattr(user, "first_name", None),
        "last_seen": int(time.time()),
    }
    save_known_users()


def is_admin(message_or_call: Any) -> bool:
    if not ADMIN_USER_ID:
        return False
    user = getattr(message_or_call, "from_user", None)
    if user is None and getattr(message_or_call, "message", None):
        user = getattr(message_or_call.message, "from_user", None)
    return bool(user and user.id == ADMIN_USER_ID)


def admin_stats_text() -> str:
    total_users = len(known_users)
    recent_day = sum(
        1 for item in known_users.values()
        if int(time.time()) - int(item.get("last_seen", 0)) <= 24 * 60 * 60
    )
    return (
        "Статистика бота:\n"
        f"Усього користувачів: {total_users}\n"
        f"Активні за 24 години: {recent_day}\n"
        f"ADMIN_USER_ID: {ADMIN_USER_ID or 'не задано'}"
    )


def acquire_single_instance_lock() -> None:
    global single_instance_socket

    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lock_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        lock_socket.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
        lock_socket.listen(1)
    except OSError as exc:
        lock_socket.close()
        raise RuntimeError(
            "Бот уже запущений в іншому вікні або фоновому процесі. "
            "Закрийте стару копію перед новим запуском."
        ) from exc

    single_instance_socket = lock_socket


def release_single_instance_lock() -> None:
    global single_instance_socket
    if single_instance_socket is not None:
        single_instance_socket.close()
        single_instance_socket = None

gemini_client = None
gemini_model = None
if HAS_GOOGLE_GENAI and GEMINI_API_KEY:
    try:
        gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini connected through google-genai: %s", GEMINI_MODEL_NAME)
    except Exception:
        logger.exception("google-genai initialization failed")

if not gemini_client and HAS_LEGACY_GEMINI and GEMINI_API_KEY:
    try:
        legacy_genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = legacy_genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info("Gemini connected through google-generativeai: %s", GEMINI_MODEL_NAME)
    except Exception:
        logger.exception("Legacy Gemini initialization failed")
elif not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is missing. The bot will try g4f fallback.")


def build_reply_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=False)
    keyboard.row(KeyboardButton("🧠 Мозковий штурм"), KeyboardButton("💬 Жива відповідь"))
    keyboard.row(KeyboardButton("🌍 Переклад"), KeyboardButton("📝 Резюме"))
    keyboard.row(KeyboardButton("😂 Жарт"), KeyboardButton("🌟 Факт"))
    keyboard.row(KeyboardButton("📤 Експорт"), KeyboardButton("⚙️ Статус"))
    keyboard.row(KeyboardButton("🧹 Очистити"), KeyboardButton("❓ Допомога"))
    return keyboard


def build_inline_menu() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("Очистити", callback_data="clear_context"),
        InlineKeyboardButton("Статус", callback_data="show_status"),
    )
    markup.row(
        InlineKeyboardButton("Експорт", callback_data="export_history"),
        InlineKeyboardButton("Допомога", callback_data="show_help"),
    )
    return markup


def build_feedback_markup() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("Добре", callback_data="feedback:up"),
        InlineKeyboardButton("Не те", callback_data="feedback:down"),
        InlineKeyboardButton("Повторити", callback_data="feedback:retry"),
    )
    return markup


def build_translate_inline() -> InlineKeyboardMarkup:
    languages = [
        ("Українська", "uk"),
        ("Англійська", "en"),
        ("Польська", "pl"),
        ("Німецька", "de"),
        ("Французька", "fr"),
        ("Іспанська", "es"),
    ]
    markup = InlineKeyboardMarkup()
    for name, code in languages:
        markup.add(InlineKeyboardButton(f"🌍 {name}", callback_data=f"translate:{code}"))
    markup.add(InlineKeyboardButton("🔎 Авто-детект", callback_data="translate:auto"))
    return markup


def build_image_actions_markup() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("🔎 Текст", callback_data="image:extract"),
        InlineKeyboardButton("🌍 Перекласти", callback_data="image:translate"),
    )
    markup.row(
        InlineKeyboardButton("🖼️ Опис", callback_data="image:describe"),
        InlineKeyboardButton("📝 Резюме", callback_data="image:summary"),
    )
    return markup


def ensure_chat_context(chat_id: int) -> list[dict[str, str]]:
    if not chat_contexts[chat_id]:
        chat_contexts[chat_id].append({"role": "system", "content": SYSTEM_PROMPT})
    return chat_contexts[chat_id]


def trim_context(chat_id: int) -> None:
    history = ensure_chat_context(chat_id)
    if len(history) <= MAX_CONTEXT_MESSAGES:
        return
    chat_contexts[chat_id] = [history[0]] + history[-(MAX_CONTEXT_MESSAGES - 1) :]


def g4f_history(chat_id: int) -> list[dict[str, str]]:
    return ensure_chat_context(chat_id)


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit].rstrip()}\n\n[Обрізано {omitted} символів, щоб запит не був завеликим.]"


def compact_history(chat_id: int) -> list[dict[str, str]]:
    history = ensure_chat_context(chat_id)
    result = []
    used_chars = 0

    for item in reversed(history[1:]):
        content = item["content"]
        if used_chars + len(content) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used_chars
            if remaining > 200:
                result.append({**item, "content": truncate_text(content, remaining)})
            break
        result.append(item)
        used_chars += len(content)

    return history[:1] + list(reversed(result))


def status_text(chat_id: int) -> str:
    history_len = max(len(chat_contexts.get(chat_id, [])) - 1, 0)
    ai_services = []
    if gemini_client or gemini_model:
        ai_services.append(f"Gemini: {GEMINI_MODEL_NAME}")
    if HAS_G4F:
        state = "увімкнений" if ENABLE_G4F_FALLBACK or not (gemini_client or gemini_model) else "вимкнений"
        ai_services.append(f"g4f: {state}")
    if not ai_services:
        ai_services.append("немає доступних AI-сервісів")

    return (
        "<b>Стан бота</b>:\n"
        f"• AI: {', '.join(ai_services)}\n"
        f"• OCR: {'доступний' if HAS_TESSERACT else 'недоступний'}\n"
        f"• Платформа: {platform.system()} {platform.release()}\n"
        f"• Повідомлень у контексті: {history_len}\n"
        f"• AI concurrency: {AI_CONCURRENCY}"
    )


def history_text(chat_id: int) -> str:
    history = ensure_chat_context(chat_id)
    visible = history[1:]
    if not visible:
        return "Історія поки порожня."
    lines = []
    for item in visible[-10:]:
        role = "Ти" if item["role"] == "user" else "Бот"
        text = item["content"].replace("\n", " ")
        lines.append(f"{role}: {text[:500]}")
    return "\n".join(lines)


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    limit = min(max(limit, 1), TELEGRAM_LIMIT)
    if not text:
        return ["Не вдалося отримати відповідь."]

    chunks = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = text.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


async def send_long_message(
    chat_id: int,
    text: str,
    reply_markup: Any = None,
    reply_to_message_id: int | None = None,
) -> None:
    chunks = split_message(text)
    for index, chunk in enumerate(chunks):
        try:
            await bot.send_message(
                chat_id,
                chunk,
                reply_markup=reply_markup if index == len(chunks) - 1 else None,
                reply_to_message_id=reply_to_message_id if index == 0 else None,
            )
        except Exception:
            logger.exception("Failed to send message chunk")
            if reply_to_message_id:
                await bot.send_message(chat_id, chunk[:TELEGRAM_LIMIT])


async def keep_typing(chat_id: int, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id, "typing")
        except Exception:
            logger.debug("Could not send typing action", exc_info=True)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


async def call_gemini(chat_id: int, prompt: str, fast: bool = False) -> str | None:
    if not gemini_client and not gemini_model:
        return None

    instruction = SYSTEM_PROMPT
    if fast:
        instruction += "\nВідповідай коротко: максимум 5-7 речень, якщо не просять більше."

    history_lines = []
    for item in compact_history(chat_id)[1:]:
        role = "Користувач" if item["role"] == "user" else "Асистент"
        history_lines.append(f"{role}: {item['content']}")

    prompt = truncate_text(prompt, MAX_USER_CHARS)
    full_prompt = (
        f"{instruction}\n\n"
        f"Історія діалогу:\n{chr(10).join(history_lines[-MAX_CONTEXT_MESSAGES:])}\n\n"
        f"Поточний запит:\n{prompt}\n\n"
        "Відповідь:"
    )

    if gemini_client:
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL_NAME,
            contents=full_prompt,
        )
        return getattr(response, "text", None)

    response = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
    return getattr(response, "text", None)


async def call_g4f(chat_id: int, prompt: str) -> str | None:
    gemini_available = bool(gemini_client or gemini_model)
    if not HAS_G4F or (gemini_available and not ENABLE_G4F_FALLBACK):
        return None
    messages = compact_history(chat_id) + [{"role": "user", "content": truncate_text(prompt, MAX_USER_CHARS)}]
    response = await asyncio.to_thread(
        g4f.ChatCompletion.create,
        model=g4f.models.default,
        messages=messages,
    )
    return str(response) if response else None


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def optimize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= MAX_IMAGE_SIDE:
        return image

    resized = image.copy()
    resized.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return resized


async def call_gemini_vision(prompt: str, image: Image.Image) -> str | None:
    if not gemini_client and not gemini_model:
        return None

    image_bytes = image_to_jpeg_bytes(image)

    if gemini_client and google_types:
        image_part = google_types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL_NAME,
            contents=[prompt, image_part],
        )
        return getattr(response, "text", None)

    legacy_image = Image.open(io.BytesIO(image_bytes))
    response = await asyncio.to_thread(
        gemini_model.generate_content,
        [prompt, legacy_image],
    )
    return getattr(response, "text", None)


async def generate_ai_response(chat_id: int, prompt: str, fast: bool = False) -> str:
    ensure_chat_context(chat_id)

    last_error = None
    async with ai_semaphore:
        try:
            reply = await asyncio.wait_for(call_gemini(chat_id, prompt, fast), AI_TIMEOUT_SECONDS)
            if reply:
                return reply.strip()
        except Exception as exc:
            last_error = exc
            logger.exception("Gemini request failed")

        try:
            reply = await asyncio.wait_for(call_g4f(chat_id, prompt), AI_TIMEOUT_SECONDS)
            if reply:
                return reply.strip()
        except Exception as exc:
            last_error = exc
            logger.exception("g4f request failed")

    if last_error and "API_KEY_INVALID" in str(last_error):
        return (
            "Gemini API key недійсний. Створіть новий ключ у Google AI Studio "
            "і замініть GEMINI_API_KEY у файлі .env."
        )

    return "Зараз не вдалося отримати відповідь від AI. Спробуйте ще раз трохи пізніше."


async def answer_with_ai(
    message: Any,
    prompt: str,
    *,
    fast: bool = False,
    save_user_message: bool = True,
) -> None:
    chat_id = message.chat.id
    ensure_chat_context(chat_id)

    async with chat_locks[chat_id]:
        now = time.time()
        if now - last_request_time[chat_id] < COOLDOWN_SECONDS:
            await bot.reply_to(
                message,
                "Зачекайте секунду перед наступним запитом.",
                reply_markup=build_reply_keyboard(),
            )
            return
        last_request_time[chat_id] = now

        prompt = truncate_text(prompt, MAX_USER_CHARS)
        if save_user_message:
            last_user_message[chat_id] = prompt

        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(keep_typing(chat_id, stop_typing))
        try:
            reply = await generate_ai_response(chat_id, prompt, fast=fast)
        finally:
            stop_typing.set()
            try:
                await typing_task
            except Exception:
                logger.debug("Typing task finished with an error", exc_info=True)

        if save_user_message:
            chat_contexts[chat_id].append({"role": "user", "content": prompt})
        chat_contexts[chat_id].append({"role": "assistant", "content": reply})
        trim_context(chat_id)

    await send_long_message(
        chat_id,
        reply,
        reply_markup=build_feedback_markup(),
        reply_to_message_id=message.message_id,
    )


async def perform_translation(chat_id: int, text: str, target_lang: str) -> str:
    prompt = (
        f"Переклади текст мовою з кодом '{target_lang}'. "
        "Збережи форматування, списки й сенс. Не додавай пояснень.\n\n"
        f"Текст:\n{text}"
    )
    return await generate_ai_response(chat_id, prompt, fast=False)


async def analyze_image(chat_id: int, image: Image.Image) -> dict[str, str | None]:
    image = optimize_image(image)
    ocr_text = None
    if HAS_TESSERACT:
        try:
            ocr_text = await asyncio.to_thread(
                pytesseract.image_to_string,
                image.convert("RGB"),
                lang="ukr+eng",
            )
            ocr_text = ocr_text.strip() or None
        except pytesseract.TesseractNotFoundError:
            logger.warning("Tesseract executable is not installed or not in PATH")
        except Exception:
            logger.exception("OCR failed")

    prompt = "Опиши це зображення українською коротко і практично."
    if ocr_text:
        prompt += f"\n\nНа зображенні розпізнано текст:\n{ocr_text}"

    try:
        caption = await asyncio.wait_for(
            call_gemini_vision(prompt, image),
            AI_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Gemini vision request failed")
        caption = None

    if not caption:
        if ocr_text:
            caption = await generate_ai_response(chat_id, prompt, fast=True)
        else:
            caption = "Не вдалося описати зображення. Для опису фото потрібен Gemini API."

    return {"ocr": ocr_text, "caption": caption}


@bot.message_handler(commands=["start"])
async def send_welcome(message: Any) -> None:
    track_user(message)
    ensure_chat_context(message.chat.id)
    await bot.send_message(
        message.chat.id,
        (
            "<b>Привіт! 👋</b> Я — твій AI-помічник у Telegram."
            "\n\n"
            "Оберіть одну з кнопок або просто напишіть запит, і я відповім швидко й емоційно."
            "\n\n"
            "<i>Підтримую фото-аналітику: надішли фото, щоб я описав його або витягнув текст.</i>"
        ),
        reply_markup=build_reply_keyboard(),
        parse_mode="HTML",
    )


@bot.message_handler(commands=["help"])
async def send_help(message: Any) -> None:
    track_user(message)
    await bot.send_message(
        message.chat.id,
        (
            "<b>Допомога та команди</b>:\n"
            "<b>/start</b> - показати основне меню\n"
            "<b>/help</b> - цю підказку\n"
            "<b>/menu</b> - відкрити інлайн-меню\n"
            "<b>/clear</b> - очистити діалог\n"
            "<b>/status</b> - дізнатися стан бота\n"
            "<b>/summary</b> - резюме діалогу\n"
            "<b>/translate</b> - почати переклад\n"
            "<b>/joke</b> - швидкий жарт\n"
            "<b>/brainstorm</b> - ідеї для творчості\n"
            "<b>/fact</b> - цікавий факт\n"
            "<b>/export</b> - зберегти історію діалогу\n\n"
            "<i>Надішліть фото — я опишу його або витягну текст.</i>"
        ),
        reply_markup=build_reply_keyboard(),
        parse_mode="HTML",
    )


@bot.message_handler(commands=["menu"])
async def show_menu(message: Any) -> None:
    track_user(message)
    await bot.send_message(
        message.chat.id,
        "<b>🔧 Меню дій</b>:\nОберіть потрібну опцію нижче.",
        reply_markup=build_inline_menu(),
        parse_mode="HTML",
    )


@bot.message_handler(commands=["clear", "reset"])
async def clear_context(message: Any) -> None:
    track_user(message)
    chat_contexts[message.chat.id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    translate_pending.pop(message.chat.id, None)
    image_analyses.pop(message.chat.id, None)
    await bot.send_message(
        message.chat.id,
        "<b>🧹 Діалог очищено.</b> Можна починати заново.",
        reply_markup=build_reply_keyboard(),
        parse_mode="HTML",
    )


@bot.message_handler(commands=["status"])
async def show_status(message: Any) -> None:
    track_user(message)
    await bot.send_message(
        message.chat.id,
        status_text(message.chat.id),
        reply_markup=build_reply_keyboard(),
        parse_mode="HTML",
    )


@bot.message_handler(commands=["stats"])
async def show_admin_stats(message: Any) -> None:
    track_user(message)
    if not is_admin(message):
        await bot.send_message(message.chat.id, "Ця команда доступна тільки власнику бота.")
        return

    await bot.send_message(message.chat.id, admin_stats_text())


@bot.message_handler(commands=["summary"])
async def send_summary(message: Any) -> None:
    track_user(message)
    prompt = "Зроби коротке резюме нашого діалогу з 3-5 пунктів."
    await answer_with_ai(message, prompt, fast=True, save_user_message=False)


@bot.message_handler(commands=["joke"])
async def send_joke(message: Any) -> None:
    track_user(message)
    await answer_with_ai(
        message,
        "Розкажи короткий добрий жарт українською.",
        fast=True,
        save_user_message=False,
    )


@bot.message_handler(commands=["brainstorm"])
async def send_brainstorm(message: Any) -> None:
    track_user(message)
    await answer_with_ai(
        message,
        "Запропонуй 7 практичних ідей для продуктивного дня.",
        fast=True,
        save_user_message=False,
    )


@bot.message_handler(commands=["fact"])
async def send_fact(message: Any) -> None:
    track_user(message)
    await answer_with_ai(
        message,
        "Дай один цікавий факт і коротко поясни, чому він цікавий.",
        fast=True,
        save_user_message=False,
    )


@bot.message_handler(commands=["translate"])
async def start_translation(message: Any) -> None:
    track_user(message)
    await bot.send_message(
        message.chat.id,
        "Оберіть мову перекладу, потім надішліть текст.",
        reply_markup=build_translate_inline(),
    )


@bot.message_handler(commands=["export"])
async def export_history(message: Any) -> None:
    track_user(message)
    chat_id = message.chat.id
    history = ensure_chat_context(chat_id)
    if len(history) <= 1:
        await bot.send_message(chat_id, "Історія поки порожня.")
        return

    text = "\n\n".join(f"{item['role']}:\n{item['content']}" for item in history[1:])
    file_obj = io.BytesIO(text.encode("utf-8"))
    file_obj.name = f"chat_{chat_id}_history.txt"
    await bot.send_document(chat_id, file_obj, caption="Експорт історії діалогу")


@bot.callback_query_handler(func=lambda call: True)
async def callback_handler(call: Any) -> None:
    track_user(call)
    chat_id = call.message.chat.id
    data = call.data

    if data == "clear_context":
        chat_contexts[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        await bot.answer_callback_query(call.id, "Історію очищено.")
        await bot.send_message(chat_id, "Готово. Починаємо з чистого аркуша.")
        return

    if data == "show_status":
        await bot.answer_callback_query(call.id)
        await bot.send_message(chat_id, status_text(chat_id), reply_markup=build_reply_keyboard())
        return

    if data == "show_help":
        await bot.answer_callback_query(call.id)
        await send_help(call.message)
        return

    if data == "export_history":
        await bot.answer_callback_query(call.id)
        await export_history(call.message)
        return

    if data.startswith("translate:"):
        translate_pending[chat_id] = data.split(":", 1)[1]
        await bot.answer_callback_query(call.id, "Мову обрано.")
        await bot.send_message(chat_id, "Надішліть текст для перекладу.")
        return

    if data.startswith("feedback:"):
        action = data.split(":", 1)[1]
        if action in {"up", "down"}:
            feedback_store[(chat_id, call.message.message_id)] = action
            await bot.answer_callback_query(call.id, "Дякую за відгук.")
            return

        if action == "retry":
            last = last_user_message.get(chat_id)
            if not last:
                await bot.answer_callback_query(call.id, "Немає останнього запиту.")
                return
            await bot.answer_callback_query(call.id, "Повторюю відповідь.")
            await answer_with_ai(call.message, last, fast=False, save_user_message=False)
            return

    if data.startswith("image:"):
        analysis = image_analyses.get(chat_id)
        if not analysis:
            await bot.answer_callback_query(call.id, "Спочатку надішліть зображення.")
            return

        action = data.split(":", 1)[1]
        ocr = analysis.get("ocr")
        caption = analysis.get("caption")

        if action == "extract":
            await bot.answer_callback_query(call.id)
            await send_long_message(chat_id, ocr or "Текст на зображенні не знайдено.")
            return

        if action == "describe":
            await bot.answer_callback_query(call.id)
            await send_long_message(chat_id, caption or "Не вдалося описати зображення.")
            return

        if action == "translate":
            await bot.answer_callback_query(call.id, "Перекладаю.")
            if not ocr:
                await bot.send_message(chat_id, "Текст на зображенні не знайдено.")
                return
            translated = await perform_translation(chat_id, ocr, "uk")
            await send_long_message(chat_id, translated, reply_markup=build_feedback_markup())
            return

        if action == "summary":
            await bot.answer_callback_query(call.id, "Роблю резюме.")
            source = ocr or caption
            if not source:
                await bot.send_message(chat_id, "Немає даних для резюме.")
                return
            summary = await generate_ai_response(chat_id, f"Стисло підсумуй:\n{source}", fast=True)
            await send_long_message(chat_id, summary, reply_markup=build_feedback_markup())
            return

    await bot.answer_callback_query(call.id, "Невідома дія.")


@bot.message_handler(content_types=["photo", "document"])
async def handle_image(message: Any) -> None:
    track_user(message)
    chat_id = message.chat.id
    ensure_chat_context(chat_id)

    file_id = None
    if message.content_type == "photo" and message.photo:
        file_id = message.photo[-1].file_id
    elif message.content_type == "document" and getattr(message, "document", None):
        mime_type = getattr(message.document, "mime_type", "") or ""
        if mime_type.startswith("image/"):
            file_id = message.document.file_id

    if not file_id:
        await bot.reply_to(message, "Надішліть фото або файл-зображення.")
        return

    await bot.send_chat_action(chat_id, "typing")
    try:
        file_info = await bot.get_file(file_id)
        file_bytes = await bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        logger.exception("Image download/open failed")
        await bot.reply_to(message, "Не вдалося завантажити або прочитати зображення.")
        return

    await bot.send_message(chat_id, "Аналізую зображення...")
    analysis = await analyze_image(chat_id, image)
    image_analyses[chat_id] = analysis

    text = analysis.get("caption") or "Опис недоступний."
    if analysis.get("ocr"):
        text += f"\n\nРозпізнаний текст:\n{analysis['ocr'][:1200]}"
    if not HAS_TESSERACT:
        text += "\n\nOCR недоступний: встановіть Tesseract і пакет pytesseract."

    await send_long_message(chat_id, text, reply_markup=build_image_actions_markup())


@bot.message_handler(content_types=["text"])
async def handle_text(message: Any) -> None:
    track_user(message)
    chat_id = message.chat.id
    text = (message.text or "").strip()
    if not text:
        await bot.reply_to(message, "Надішліть текстове повідомлення.")
        return

    button_prompts = {
        "Історія": None,
        "Очистити": None,
        "Резюме": "Зроби коротке резюме нашого діалогу з 3-5 пунктів.",
        "Ідея": "Запропонуй 5 ідей для корисного запиту до AI-бота.",
        "Жарт": "Розкажи короткий добрий жарт українською.",
        "Переклад": None,
        "Мозковий штурм": "Згенеруй 10 ідей для нового маленького проєкту.",
        "Факт": "Дай один цікавий факт і коротко поясни його.",
        "Допомога": None,
        "Статус": None,
    }

    if text == "Історія":
        await send_long_message(chat_id, history_text(chat_id), reply_markup=build_reply_keyboard())
        return

    if text == "Очистити":
        await clear_context(message)
        return

    if text == "Переклад":
        await start_translation(message)
        return

    if text == "Допомога":
        await send_help(message)
        return

    if text == "Статус":
        await show_status(message)
        return

    if text in button_prompts and button_prompts[text]:
        await answer_with_ai(message, button_prompts[text], fast=True, save_user_message=False)
        return

    pending_lang = translate_pending.pop(chat_id, None)
    if pending_lang:
        await bot.send_chat_action(chat_id, "typing")
        translated = await perform_translation(chat_id, text, pending_lang)
        await send_long_message(
            chat_id,
            translated,
            reply_markup=build_feedback_markup(),
            reply_to_message_id=message.message_id,
        )
        return

    await answer_with_ai(message, text, fast=True)


@bot.message_handler(func=lambda message: True, content_types=["audio", "voice", "video", "sticker"])
async def unsupported_message(message: Any) -> None:
    track_user(message)
    await bot.reply_to(
        message,
        "Поки що я найкраще працюю з текстом і зображеннями.",
        reply_markup=build_reply_keyboard(),
    )


async def remove_webhook_if_any() -> None:
    if not BOT_TOKEN:
        return

    def delete_webhook() -> None:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
        response = requests.post(url, timeout=10)
        response.raise_for_status()

    try:
        await asyncio.to_thread(delete_webhook)
        logger.info("Webhook removed or already empty")
    except Exception:
        logger.exception("Could not remove webhook")


async def main() -> None:
    logger.info("Starting bot")
    acquire_single_instance_lock()
    try:
        load_known_users()
        await remove_webhook_if_any()
        while True:
            try:
                logger.info("Polling started")
                await bot.polling(non_stop=True, timeout=60, request_timeout=90)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if "409" in str(exc) or "Conflict" in str(exc):
                    logger.error(
                        "Telegram 409 Conflict: another bot instance is using getUpdates. "
                        "Stop extra main1.py processes."
                    )
                    await asyncio.sleep(10)
                else:
                    logger.exception("Polling failed. Restarting in 5 seconds")
                    await asyncio.sleep(5)
    finally:
        release_single_instance_lock()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped")
