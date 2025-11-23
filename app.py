import os
import tempfile
import logging
from flask import Flask
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from utils.pdf_reader import extract_pdf_text
from utils.qa_engine import answer_questions_from_html

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Хранилище лекций (только один PDF на всех пользователей)
LECTURES = {}

BOT_TOKEN = os.getenv("BOT_TOKEN")

@app.route("/")
def home():
    return "Bot is running on Render!"

# ===== Telegram bot =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Загружайте PDF-лекцию, затем отправляйте HTML-код теста."
    )

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    doc = update.message.document
    if not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("Пожалуйста, отправьте PDF файл.")
        return

    file = await doc.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        file_path = temp.name
        await file.download_to_drive(file_path)
        text = extract_pdf_text(file_path)

        # Очищаем предыдущие сессии
        LECTURES.clear()
        LECTURES[chat_id] = text

    await update.message.reply_text(
        "PDF загружен! Старые данные очищены. Теперь отправьте HTML-код теста."
    )

async def handle_html(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id not in LECTURES:
        await update.message.reply_text("Сначала загрузите PDF-лекцию!")
        return

    html = update.message.text
    pdf_text = LECTURES[chat_id]
    answers = answer_questions_from_html(pdf_text, html)
    await update.message.reply_text(answers)

def run_bot():
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_html))
    application.run_polling()

if __name__ == "__main__":
    run_bot()
