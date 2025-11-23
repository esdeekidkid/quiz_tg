import os
from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes
from utils.pdf_reader import extract_pdf_text
from utils.qa_engine import answer_questions_from_html

app = Flask(__name__)

LECTURES = {}

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # https://your-render-domain.onrender.com/webhook

telegram_app = ApplicationBuilder().token(BOT_TOKEN).build()


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправьте PDF, затем HTML-код теста."
    )


async def pdf_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    doc = update.message.document

    if not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("Отправьте PDF файл.")
        return

    file = await doc.get_file()
    file_path = f"/tmp/{doc.file_name}"
    await file.download_to_drive(file_path)

    text = extract_pdf_text(file_path)
    LECTURES[chat_id] = text

    await update.message.reply_text("PDF загружен. Теперь отправьте HTML теста.")


async def html_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id not in LECTURES:
        await update.message.reply_text("Сначала отправьте PDF.")
        return

    html = update.message.text
    pdf_text = LECTURES[chat_id]

    answer = answer_questions_from_html(pdf_text, html)
    await update.message.reply_text(answer)


telegram_app.add_handler(CommandHandler("start", start_handler))
telegram_app.add_handler(MessageHandler(filters.Document.PDF, pdf_handler))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, html_handler))


@app.post("/webhook")
async def webhook():
    update = Update.de_json(request.json, telegram_app.bot)
    await telegram_app.process_update(update)
    return "OK", 200


@app.get("/")
def home():
    return "Bot is running!", 200


if __name__ == "__main__":
    telegram_app.run_webhook(
        listen="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        url_path="/webhook",
        webhook_url=os.getenv("WEBHOOK_URL"),
    )
