import os
from flask import Flask, request
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from utils.pdf_reader import extract_pdf_text
from utils.qa_engine import answer_questions_from_html

app = Flask(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

application = ApplicationBuilder().token(BOT_TOKEN).build()

LECTURES = {}


# --- Telegram handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправьте PDF, затем HTML теста."
    )


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    doc = update.message.document

    if not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("Нужен PDF файл.")
        return

    file = await doc.get_file()
    path = "/tmp/current.pdf"
    await file.download_to_drive(path)

    text = extract_pdf_text(path)

    # очищаем память
    LECTURES.clear()
    LECTURES[chat_id] = text

    await update.message.reply_text("PDF получен. Теперь отправьте HTML.")


async def handle_html(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id

    if chat_id not in LECTURES:
        await update.message.reply_text("Сначала загрузите PDF.")
        return

    html = update.message.text
    pdf_text = LECTURES[chat_id]

    answer = answer_questions_from_html(pdf_text, html)
    await update.message.reply_text(answer)


# --- Register handlers ---
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_html))


# --- Flask webhook route ---
@app.post("/webhook")
async def webhook():
    update = Update.de_json(request.get_json(), application.bot)
    await application.process_update(update)
    return "ok"


@app.get("/")
def home():
    return "Bot running"


# --- Launch bot ---
if __name__ == "__main__":
    import asyncio

    async def setup():
        await application.bot.delete_webhook()
        await application.bot.set_webhook(f"{WEBHOOK_URL}/webhook")
        await application.initialize()
        await application.start()

    asyncio.run(setup())
    app.run(host="0.0.0.0", port=10000)
