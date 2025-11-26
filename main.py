import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import pdfplumber
from bs4 import BeautifulSoup
import re
import json

app = FastAPI(title="Quiz Helper API")

# Подключаем папки templates и static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Глобальное хранилище для сессии (в реальном проекте используйте Redis или БД)
SESSION_STORAGE = {}

# --- Утилиты ---

def normalize_text(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip().lower()

def excerpt_around(text, idx, length=120):
    if not text:
        return ""
    start = max(0, idx - length // 2)
    end = min(len(text), idx + length // 2)
    return text[start:end].replace("\n", " ").replace("\r", " ")

def find_definition_in_lecture(lecture, term):
    L = lecture
    term_escaped = re.escape(term)
    # Ищем "TERM - это", "TERM — это", "TERM это", "это TERM" и т.д.
    patterns = [
        r"([А-Яа-яA-Za-z0-9\-\s]{1,80})[\-—]\s*это",
        rf"({term_escaped})\s*(?:[:\-—])\s*это",
        rf"{term_escaped}\s+представляет собой",
        rf"это\s+({term_escaped})",
    ]
    for pattern in patterns:
        match = re.search(pattern, L, re.IGNORECASE)
        if match:
            return {
                "found": True,
                "match": match.group(0),
                "snippet": excerpt_around(L, match.start())
            }
    return {"found": False}

def score_option_by_lecture(lecture, option):
    L = normalize_text(lecture)
    opt = normalize_text(option)

    score = 0
    snippets = []

    # точное вхождение опции
    exact_matches = re.findall(re.escape(opt), L)
    exact_count = len(exact_matches)
    if exact_count > 0:
        score += 3 * (1 + exact_count)**0.5  # Логарифмическое начисление
        first_match_idx = L.find(opt)
        if first_match_idx != -1:
            snippets.append({"why": "exact", "excerpt": excerpt_around(L, first_match_idx)})

    # фраза типа "opt представляет собой" или "opt - это"
    def_pattern = rf"{re.escape(opt)}\s+(представляет собой|является|это|характеризуется|обозначает|означает)"
    if re.search(def_pattern, lecture, re.IGNORECASE):
        score += 3
        match = re.search(def_pattern, lecture, re.IGNORECASE)
        snippets.append({"why": "definition", "excerpt": excerpt_around(lecture, match.start())})

    # пересечение слов
    opt_words = set(opt.split())
    if opt_words:
        matched_words = len(opt_words.intersection(set(L.split())))
        ratio = matched_words / len(opt_words)
        score += ratio * 2
        if ratio > 0:
            snippets.append({"why": "words", "matched": f"{matched_words}/{len(opt_words)}"})

    # небольшой бонус за длинную опцию, если она встречается
    if len(opt) > 30 and exact_count > 0:
        score += 0.5

    return {"score": score, "snippets": snippets}

def detect_question_type(qtext):
    q = normalize_text(qtext)
    single_markers = ['какое из', 'какой из', 'как называется', 'что из', 'выберите один', 'выберите', 'какое слово пропущено']
    for marker in single_markers:
        if marker in q:
            return 'single'

    multi_markers = ['какие', 'перечисл', 'классификация', 'входят в', 'относятся', 'какие действия', 'назовите', 'перечислите', 'признаков', 'включают']
    for marker in multi_markers:
        if marker in q:
            return 'multi'

    if 'единиц' in q or 'единицы измерения' in q or 'единицы' in q:
        return 'units'

    if re.search(r'(какое слово пропущено|какое слово|впишите|введите|короткий ответ|ответ)', qtext, re.IGNORECASE):
        return 'short'

    if 'что' in q or 'определ' in q:
        return 'single'

    return 'single'

def parse_html_quiz(html):
    # --- Добавим обработку исключений ---
    try:
        # Проверим, не является ли html уже декодированным юникодом, но с тегами в виде \uXXXX
        # Это редкий случай, но возможный. BeautifulSoup обычно справляется сам.
        soup = BeautifulSoup(html, 'html.parser')
    except Exception as e:
        print(f"Error parsing HTML with BeautifulSoup: {e}") # Логирование ошибки
        return [] # Возвращаем пустой список, если не смогли распарсить

    questions = []

    # Найдём *все* элементы с классом 'que' - это основной контейнер вопроса в Moodle
    que_elements = soup.find_all(class_='que')

    for el in que_elements:
        q = {}
        # Вопрос: находим текст вопроса внутри .qtext
        qtext_el = el.find(class_='qtext')
        if qtext_el:
            # Убираем label и input из текста вопроса, если они есть
            for tag in qtext_el.find_all(['label', 'input']):
                tag.decompose()
            q['question'] = qtext_el.get_text(strip=True).replace('\n', ' ')
        else:
            q['question'] = f"Вопрос {len(questions) + 1}" # Запасной вариант

        # Опции: находим внутри .answer или внутри .ablock
        opts = []
        # Сначала пробуем .answer
        answer_divs = el.find_all(class_='answer')
        for div in answer_divs:
            # Ищем label с data-region="answer-label" - это современный способ в Moodle
            labels = div.find_all(attrs={'data-region': 'answer-label'})
            for label in labels:
                opt_text = label.get_text(strip=True).replace('\n', ' ')
                if opt_text:
                    opts.append(opt_text)
            # Ищем старые label
            for label in div.find_all('label'):
                # Проверяем, не является ли он частью .qtext или другого системного блока
                if not label.find_parent(class_='qtext'):
                    opt_text = label.get_text(strip=True).replace('\n', ' ')
                    if opt_text and opt_text not in opts:
                        opts.append(opt_text)

        # Если не нашли в .answer, пробуем внутри .ablock
        if not opts:
             ablock = el.find('fieldset', class_='ablock')
             if ablock:
                 labels = ablock.find_all(attrs={'data-region': 'answer-label'})
                 for label in labels:
                     opt_text = label.get_text(strip=True).replace('\n', ' ')
                     if opt_text:
                         opts.append(opt_text)
                 for label in ablock.find_all('label'):
                     if not label.find_parent(class_='qtext'):
                         opt_text = label.get_text(strip=True).replace('\n', ' ')
                         if opt_text and opt_text not in opts:
                             opts.append(opt_text)

        q['options'] = list(set(opts)) # dedupe

        # Проверим, является ли коротким ответом
        # Проверим input type=text, textarea, или input с именем, содержащим _answer
        is_short = False
        qtext_part = el.find(class_='qtext')
        if qtext_part:
             # Проверяем input/textarea внутри .qtext
             if qtext_part.find(['input', 'textarea'], attrs={'type': 'text'}):
                 is_short = True
             # Проверяем input с именем, содержащим _answer
             if qtext_part.find(['input', 'textarea'], attrs={'name': lambda x: x and '_answer' in x}):
                 is_short = True

        # Проверяем в остальной части элемента вопроса
        if not is_short:
            if el.find(['input', 'textarea'], attrs={'type': 'text'}):
                 is_short = True
            if el.find(['input', 'textarea'], attrs={'name': lambda x: x and '_answer' in x}):
                 is_short = True

        q['is_short'] = is_short
        questions.append(q)

    # Если не нашли по классам, ищем вручную по эвристике (менее надёжно)
    if not questions:
        # Ищем все <p> внутри .content или .formulation, предполагая, что это вопросы
        # Это менее надёжно
        content_blocks = soup.find_all(class_='content') or soup.find_all(class_='formulation')
        for block in content_blocks:
            question_text = block.find(class_='qtext')
            if question_text:
                qtext = question_text.get_text(strip=True).replace('\n', ' ')
                if qtext:
                    opts = []
                    # Ищем опции
                    answer_divs = block.find_all(class_='answer')
                    for div in answer_divs:
                        labels = div.find_all('label')
                        for label in labels:
                            opt_text = label.get_text(strip=True).replace('\n', ' ')
                            if opt_text:
                                opts.append(opt_text)
                    # Заполняем q
                    q = {
                        "question": qtext,
                        "options": list(set(opts)),
                        "is_short": bool(block.find(['input', 'textarea'], attrs={'type': 'text'}))
                    }
                    questions.append(q)

    return questions


# --- Маршруты ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/extract-text-from-pdf/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Файл должен быть PDF")

    content = await file.read()

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: # Проверяем, что текст не None или пустой
                    text += page_text + " "
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при извлечении текста: {str(e)}")

    # Сохраняем текст в сессию (упрощённо, используя IP)
    # В реальности используйте JWT токены или сессии
    session_key = "default" # Упрощённо, можно использовать request.client.host
    SESSION_STORAGE[session_key] = text

    return {
        "text": text,
        "length": len(text),
        "snippet": text[:200]
    }

# --- Модель для парсинга HTML ---
class ParseQuizHTML(BaseModel):
    html: str

# --- Новый маршрут для парсинга HTML ---
@app.post("/api/parse-quiz-html/")
async def parse_quiz_html(data: ParseQuizHTML): # Принимаем JSON из тела запроса
    html = data.html
    if not html:
        raise HTTPException(status_code=400, detail="Поле 'html' отсутствует или пусто.")

    # --- Попробуем распарсить HTML ---
    try:
        questions = parse_html_quiz(html)
    except Exception as e:
        # Если parse_html_quiz выбросит исключение (хотя мы его и обернули внутри), ловим здесь
        print(f"Error in parse_html_quiz: {e}") # Логирование ошибки
        raise HTTPException(status_code=500, detail=f"Ошибка при парсинге HTML теста: {str(e)}")

    if not questions:
        raise HTTPException(status_code=400, detail="В HTML не найдено ни одного вопроса. Убедитесь, что HTML содержит правильную структуру теста (например, div с классом 'que').")

    return {"ok": True, "questions": questions}


# --- Обновлённый маршрут для обработки квиза (принимает уже распарсенные вопросы) ---
class ProcessQuizData(BaseModel):
    questions: List[dict] # Список вопросов в формате, как возвращается из /api/parse-quiz-html/
    lecture_text: str

@app.post("/api/process-quiz/")
async def process_quiz(data: ProcessQuizData): # <-- ИСПРАВЛЕНО: принимаем модель из тела
    # --- Валидация данных вручную ---
    questions = data.questions
    lecture_text = data.lecture_text

    if not lecture_text:
        raise HTTPException(status_code=400, detail="Поле 'lecture_text' отсутствует или пусто. Сначала загрузите PDF-лекцию и убедитесь, что она успешно обработалась.")

    if not questions:
        raise HTTPException(status_code=400, detail="Поле 'questions' отсутствует или пусто. Сначала распарсите HTML теста.")

    results = []

    for q in questions:
        qtext = q.get("question", "")
        opts = q.get("options", [])
        is_short = q.get("is_short", False)
        # Определяем тип вопроса на основе текста
        qtype = detect_question_type(qtext)

        if is_short or qtype == 'short':
            lec = lecture_text
            # Ищем определения
            def_regex = r"([А-Яа-яЁёA-Za-z0-9 \-]{2,80})\s*[—\-:]\s*это"
            match = re.search(def_regex, lec, re.IGNORECASE)
            found = None
            if match:
                candidate = match.group(1).strip()
                if len(candidate.split()) <= 6: # Ограничение на длину
                    found = {"answer": candidate, "excerpt": excerpt_around(lec, match.start())}

            if not found:
                 # Попробуем "TERM это"
                 alt_regex = r"([А-Яа-яЁёA-Za-z0-9 \-]{2,60})\s+это\s+"
                 alt_match = re.search(alt_regex, lec, re.IGNORECASE)
                 if alt_match:
                     candidate = alt_match.group(1).strip()
                     found = {"answer": candidate, "excerpt": excerpt_around(lec, alt_match.start())}

            results.append({
                "question": qtext,
                "type": "short",
                "answer": found["answer"] if found else "",
                "excerpt": found["excerpt"] if found else "",
            })
            continue

        # Обработка вопросов с опциями
        scored = []
        for opt in opts:
            score_result = score_option_by_lecture(lecture_text, opt)
            scored.append({
                "option": opt,
                "score": score_result["score"],
                "snippets": score_result["snippets"]
            })

        max_score = max([s["score"] for s in scored], default=1)
        for s in scored:
            s["norm"] = round(s["score"] / max_score, 3) if max_score > 0 else 0

        selected = []
        if qtype == 'single':
            top = max(scored, key=lambda x: x["score"], default=None)
            if top:
                selected = [{"option": top["option"], "score": top["norm"], "snippets": top["snippets"]}]
        elif qtype == 'units':
            selected = [s for s in scored if s["norm"] >= 0.15]
            selected.sort(key=lambda x: x["norm"], reverse=True)
        else: # multi
            candidates = [s for s in scored if s["norm"] >= 0.55]
            if not candidates:
                fallback = [s for s in scored if s["norm"] >= 0.25]
                selected = fallback
            else:
                selected = candidates
            selected.sort(key=lambda x: x["norm"], reverse=True)

        results.append({
            "question": qtext,
            "type": qtype,
            "options": [{"option": s["option"], "norm": s["norm"], "snippets": s["snippets"]} for s in scored],
            "selected": [{"option": s["option"], "score": s["score"], "snippets": s["snippets"]} for s in selected]
        })

    return {"ok": True, "results": results}

# Запуск: uvicorn main:app --host 0.0.0.0 --port 10000