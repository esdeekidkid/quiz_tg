# ============================
# main.py — Полностью исправленный
# ============================

import io
import re
import pdfplumber
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from bs4 import BeautifulSoup


# ----------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ СЕРВЕРА
# ----------------------------------------------------------

app = FastAPI(title="Quiz Helper")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SESSION_STORAGE = {}   # Хранит PDF-текст (временно, без базы)


# ----------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def excerpt_around(text, idx, length=120):
    start = max(0, idx - length // 2)
    end = min(len(text), idx + length // 2)
    return text[start:end].replace("\n", " ").replace("\r", " ")


def score_option_by_lecture(lecture: str, option: str):
    L = normalize_text(lecture)
    opt = normalize_text(option)

    score = 0
    snippets = []

    # 1) Прямое совпадение
    idx = L.find(opt)
    if idx != -1:
        score += 3
        snippets.append({"why": "exact", "excerpt": excerpt_around(L, idx)})

    # 2) "... представляет собой", "... является ..."
    def_pattern = rf"{re.escape(opt)}\s+(представляет собой|является|это)"
    match = re.search(def_pattern, lecture, re.IGNORECASE)
    if match:
        score += 3
        snippets.append({"why": "definition", "excerpt": excerpt_around(lecture, match.start())})

    # 3) Пересечение слов
    opt_words = set(opt.split())
    L_words = set(L.split())
    inter = len(opt_words & L_words)

    if inter > 0:
        score += (inter / len(opt_words)) * 2
        snippets.append({"why": "word-match", "matched": f"{inter}/{len(opt_words)}"})

    return {"score": score, "snippets": snippets}


def detect_question_type(text: str) -> str:
    t = normalize_text(text)

    if any(x in t for x in [
        "какое слово", "введите", "впишите", "короткий ответ"
    ]):
        return "short"

    if any(x in t for x in [
        "классификация", "перечислите", "какие", "относятся"
    ]):
        return "multi"

    if "единиц" in t or "единицы измерения" in t:
        return "units"

    return "single"


def parse_html_quiz(html: str):
    soup = BeautifulSoup(html, "html.parser")

    questions = []

    que_blocks = soup.find_all(class_="que")
    for block in que_blocks:
        # Текст вопроса
        qtext_el = block.find(class_="qtext")
        qtext = qtext_el.get_text(" ", strip=True) if qtext_el else "Вопрос"

        # Варианты
        opts = []
        answer_divs = block.find_all(class_="answer")

        for ad in answer_divs:
            labels = ad.find_all(attrs={"data-region": "answer-label"})
            for lb in labels:
                t = lb.get_text(strip=True)
                if t:
                    opts.append(t)

            for lb in ad.find_all("label"):
                t = lb.get_text(strip=True)
                if t and t not in opts:
                    opts.append(t)

        # Удаление дубликатов
        opts = list(dict.fromkeys(opts))

        questions.append({
            "question": qtext,
            "options": opts,
            "is_short": bool(block.find("input", {"type": "text"}))
        })

    return questions


# ----------------------------------------------------------
# MODELS
# ----------------------------------------------------------

class HtmlPayload(BaseModel):
    html: str


class ProcessQuizData(BaseModel):
    questions: List[dict]
    lecture_text: str


# ----------------------------------------------------------
# ROUTES
# ----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Загружаем PDF и извлекаем текст ---
@app.post("/api/extract-text-from-pdf/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Файл должен быть PDF.")

    content = await file.read()

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка PDF: {e}")

    SESSION_STORAGE["default"] = text

    return {
        "text": text,
        "length": len(text),
        "snippet": text[:300]
    }


# --- Парсим HTML теста ---
@app.post("/api/parse-quiz-html/")
async def parse_quiz_html(payload: HtmlPayload):
    html = payload.html

    if not html:
        raise HTTPException(status_code=400, detail="Поле 'html' пустое")

    try:
        questions = parse_html_quiz(html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при парсинге HTML: {e}")

    if not questions:
        raise HTTPException(status_code=400, detail="В HTML не найдено ни одного вопроса")

    return {"ok": True, "questions": questions}


# --- Проводим анализ: находим правильные ответы ---
@app.post("/api/process-quiz/")
async def process_quiz(payload: ProcessQuizData):
    questions = payload.questions
    lecture_text = payload.lecture_text

    if not lecture_text:
        raise HTTPException(status_code=400, detail="lecture_text отсутствует")

    if not questions:
        raise HTTPException(status_code=400, detail="questions отсутствуют")

    results = []

    for q in questions:
        qtext = q.get("question", "")
        opts = q.get("options", [])
        is_short = q.get("is_short", False)

        qtype = detect_question_type(qtext)

        # ------------------------------
        # короткий ответ
        # ------------------------------
        if is_short or qtype == "short":
            # ищем "... — это" или "... - это"
            m = re.search(r"([А-Яа-яA-Za-z0-9 \-]{3,60})\s*[—\-:]\s*это", lecture_text)
            if m:
                ans = m.group(1).strip()
                results.append({
                    "question": qtext,
                    "type": "short",
                    "answer": ans,
                    "excerpt": excerpt_around(lecture_text, m.start())
                })
                continue

            results.append({
                "question": qtext,
                "type": "short",
                "answer": "",
                "excerpt": ""
            })
            continue

        # ------------------------------
        # вопросы с вариантами
        # ------------------------------
        scored = []
        for opt in opts:
            sc = score_option_by_lecture(lecture_text, opt)
            scored.append({
                "option": opt,
                "score": sc["score"],
                "snippets": sc["snippets"]
            })

        max_score = max([s["score"] for s in scored], default=1)

        for s in scored:
            s["norm"] = round(s["score"] / max_score, 3)

        if qtype == "single":
            selected = [max(scored, key=lambda x: x["score"])]

        elif qtype == "units":
            selected = [s for s in scored if s["norm"] >= 0.15]

        else:  # multi
            selected = [s for s in scored if s["norm"] >= 0.55]
            if not selected:
                selected = [s for s in scored if s["norm"] >= 0.3]

        selected.sort(key=lambda x: x["norm"], reverse=True)

        results.append({
            "question": qtext,
            "type": qtype,
            "options": scored,
            "selected": selected
        })

    return {"ok": True, "results": results}
