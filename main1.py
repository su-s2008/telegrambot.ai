import os
import platform
import asyncio
from telebot.async_telebot import AsyncTeleBot

# Спроба імпорту Google Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ModuleNotFoundError:
    HAS_GEMINI = False

# Спроба імпорту безкоштовного g4f
try:
    import g4f
    HAS_G4F = True
except ModuleNotFoundError:
    HAS_G4F = False

# --- КОНФІГУРАЦІЯ ---
BOT_TOKEN = ""
GEMINI_API_KEY = "".strip()

bot = AsyncTeleBot(BOT_TOKEN)
chat_contexts = {}
MAX_CONTEXT_LEN = 11  # 1 системний промт + 10 повідомлень історії

# --- ІНІЦІАЛІЗАЦІЯ ОФІЦІЙНОГО GEMINI ---
gemini_model = None
if HAS_GEMINI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("[+] Офіційний стабільний ШI від Google підключено!")
    except Exception as e:
        print(f"[-] Помилка ініціалізації Gemini API: {e}")

if not gemini_model:
    print("[!] Працюємо через безкоштовні лінії g4f (Автоматичний вибір провайдера).")

# --- ОБРОБКА ПОВІДОМЛЕНЬ БОТА ---
@bot.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    await bot.reply_to(message, "Привіт! Я ваш асистент. Напишіть мені щось, і я відповім за допомогою ШІ.")

@bot.message_handler(commands=['clear'])
async def clear_context(message):
    chat_id = message.chat.id
    if chat_id in chat_contexts:
        chat_contexts[chat_id] = []
    await bot.reply_to(message, "🧹 Історію нашого діалогу успішно очищено! Про що поговоримо?")

@bot.message_handler(func=lambda message: True)
async def handle_message(message):
    user_text = message.text
    chat_id = message.chat.id

    # Створюємо історію діалогу, якщо її немає
    if chat_id not in chat_contexts:
        chat_contexts[chat_id] = []

    # Додаємо повідомлення користувача в історію
    chat_contexts[chat_id].append({"role": "user", "content": user_text})

    # Обрізаємо контекст, якщо він завеликий
    if len(chat_contexts[chat_id]) > MAX_CONTEXT_LEN:
        chat_contexts[chat_id] = chat_contexts[chat_id][-MAX_CONTEXT_LEN:]

    # Надсилаємо статус "друкує..."
    await bot.send_chat_action(chat_id, 'typing')

    reply_text = ""

    # Спроба 1: Використовуємо офіційний Gemini (якщо ключ працює)
    if gemini_model:
        try:
            # Перетворюємо контекст у формат для Gemini або передаємо просто текст
            response = gemini_model.generate_content(user_text)
            reply_text = response.text
        except Exception as e:
            print(f"[-] Помилка офіційного Gemini: {e}. Перемикаюсь на g4f...")

    # Спроба 2: Резервний безкоштовний g4f (якщо Gemini не відпрацював або немає ключа)
    if not reply_text and HAS_G4F:
        try:
            # Викликаємо БЕЗ вказання конкретного 'Blackbox', щоб уникнути помилки атрибута
            response = g4f.ChatCompletion.create(
                model=g4f.models.default,
                messages=chat_contexts[chat_id]
            )
            reply_text = response
        except Exception as e:
            reply_text = f"Вибачте, виникла помилка ШІ: {e}"
    elif not reply_text:
        reply_text = "Не вдалося підключити жоден сервіс ШІ."

    # Додаємо відповідь ШІ в історію для контексту
    chat_contexts[chat_id].append({"role": "assistant", "content": reply_text})

    # Надсилаємо відповідь користувачу в Telegram
    try:
        await bot.reply_to(message, reply_text)
    except Exception as e:
        print(f"[-] Помилка відправки повідомлення: {e}")

# --- ЗАПУСК БОТА ---
if __name__ == "__main__":
    print("[RUN] Бот успішно запущений в асинхронному режимі!")
    asyncio.run(bot.polling(none_stop=True))