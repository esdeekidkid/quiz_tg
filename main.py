import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pdfplumber
from bs4 import BeautifulSoup
import re
import json

app = FastAPI(title="Quiz Helper API")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем папки templates и static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Глобальное хранилище для сессии (в реальном проекте используйте Redis или БД)
SESSION_STORAGE = {}

# --- Pydantic модели для валидации ---
class QuizHtmlRequest(BaseModel):
    html: str = Field(..., min_length=1, description="HTML код теста")

class QuestionData(BaseModel):
    question: str
    options: List[str] = []
    is_short: bool = False

class ProcessQuizRequest(BaseModel):
    questions: List[Dict[str, Any]]
    lecture_text: str = Field(..., min_length=1)

# --- Утилиты ---

def normalize_text(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip().lower()

def excerpt_around(text, idx, length=250):
    if not text:
        return ""
    start = max(0, idx - length // 2)
    end = min(len(text), idx + length // 2)
    return text[start:end].replace("\n", " ").replace("\r", " ")

def find_definition_in_lecture(lecture, term):
    L = lecture
    term_escaped = re.escape(term)
    patterns = [
        rf"(?<!\w){term_escaped}\s*(?:[\-—]\s*это|[\-—]\s*является|[\-—]\s*означает)\s+([^.!?]+)",
        rf"(?<!\w){term_escaped}\s+(?:представляет собой|характеризуется как|обозначает)\s+([^.!?]+)",
        rf"это\s+{term_escaped}\s+(?:является|означает|представляет собой)\s+([^.!?]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, L, re.IGNORECASE)
        if match:
            definition = match.group(1).strip()
            full_match_start = match.start()
            sent_start = L.rfind('.', 0, full_match_start) + 1
            sent_end = L.find('.', full_match_start)
            if sent_end == -1:
                sent_end = L.find('!', full_match_start)
                if sent_end == -1:
                    sent_end = L.find('?', full_match_start)
                if sent_end == -1:
                    sent_end = len(L)
            snippet = L[sent_start:sent_end].strip()
            return {
                "found": True,
                "match": match.group(0),
                "definition": definition,
                "snippet": excerpt_around(L, full_match_start)
            }
    return {"found": False}

def score_option_by_lecture(lecture, option, question=""):
    L = normalize_text(lecture)
    opt = normalize_text(option)
    q = normalize_text(question)

    score = 0
    snippets = []

    context_keywords = []
    if 'эквивалентной' in q and 'дозы' in q:
        context_keywords.extend(['эквивалентной', 'дозы'])
    if 'эффективной' in q and 'дозы' in q:
        context_keywords.extend(['эффективной', 'дозы'])

    exact_matches = re.findall(re.escape(opt), L)
    exact_count = len(exact_matches)
    if exact_count > 0:
        base_score = 3 * (1 + exact_count)**0.5
        opt_has_context = any(ctx_word in opt for ctx_word in context_keywords)
        if opt_has_context:
            base_score *= 1.5
        score += base_score
        first_match_idx = L.find(opt)
        if first_match_idx != -1:
            snippets.append({"why": f"exact{'_context' if opt_has_context else ''}", "excerpt": excerpt_around(L, first_match_idx)})

    def_patterns = [
        rf"{re.escape(opt)}\s+(представляет собой|является|это|характеризуется|обозначает|означает)",
        rf"{re.escape(opt)}\s+([^-—.!?]*?)(?:[\-—]\s*это|[\-—]\s*является|[\-—]\s*означает)"
    ]
    for pat in def_patterns:
        match_iter = re.finditer(pat, lecture, re.IGNORECASE)
        for match in match_iter:
            match_text = match.group(0)
            match_has_q_word = any(word in normalize_text(match_text) for word in q.split())
            bonus = 3
            if match_has_q_word:
                bonus *= 1.2
            score += bonus
            snippets.append({"why": f"definition{'_q_match' if match_has_q_word else ''}", "excerpt": excerpt_around(lecture, match.start())})

    opt_words = set(opt.split())
    if opt_words:
        matched_words = len(opt_words.intersection(set(L.split())))
        ratio = matched_words / len(opt_words)
        score += ratio * 2
        if ratio > 0:
            snippets.append({"why": "words", "matched": f"{matched_words}/{len(opt_words)}"})

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
    soup = BeautifulSoup(html, 'html.parser')
    questions = []

    que_elements = soup.find_all(class_='que')

    for el in que_elements:
        q = {}
        qtext_el = el.find(class_='qtext')
        if qtext_el:
            for tag in qtext_el.find_all(['label', 'input']):
                tag.decompose()
            q['question'] = qtext_el.get_text(strip=True).replace('\n', ' ')
        else:
            q['question'] = f"Вопрос {len(questions) + 1}"

        opts = []
        answer_divs = el.find_all(class_='answer')
        for div in answer_divs:
            labels = div.find_all(attrs={'data-region': 'answer-label'})
            for label in labels:
                opt_text = label.get_text(strip=True).replace('\n', ' ')
                if opt_text:
                    opts.append(opt_text)
            for label in div.find_all('label'):
                if not label.find_parent(class_='qtext'):
                    opt_text = label.get_text(strip=True).replace('\n', ' ')
                    if opt_text and opt_text not in opts:
                        opts.append(opt_text)

        q['options'] = list(set(opts))
        q['is_short'] = bool(el.find('input', type='text')) or 'shortanswer' in el.get('class', [])
        questions.append(q)

    if not questions:
        all_ps = soup.find_all('p')
        all_inputs = soup.find_all('input', type='radio') + soup.find_all('input', type='checkbox')
        for p in all_ps:
            if p.find('input'):
                continue
            question_text = p.get_text(strip=True).replace('\n', ' ')
            if question_text:
                options = []
                for inp in all_inputs:
                    label = soup.find('label', attrs={'for': inp.get('id')})
                    if label:
                        opt_text = label.get_text(strip=True).replace('\n', ' ')
                        if opt_text:
                            options.append(opt_text)
                questions.append({
                    "question": question_text,
                    "options": list(set(options)),
                    "is_short": bool(soup.find('input', type='text'))
                })
                break

    return questions


# --- Маршруты ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/extract-text-from-pdf/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Файл должен быть PDF")
    
    # Проверка размера файла (макс 10MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл слишком большой (максимум 10MB)")

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при извлечении текста: {str(e)}")

    session_key = "default"
    SESSION_STORAGE[session_key] = text

    return {
        "text": text,
        "length": len(text),
        "snippet": text[:200]
    }

@app.post("/api/parse-quiz-html/")
async def parse_quiz_html(data: QuizHtmlRequest):
    html = data.html
    if not html:
        raise HTTPException(status_code=400, detail="Поле 'html' отсутствует или пусто.")

    try:
        questions = parse_html_quiz(html)
    except Exception as e:
        print(f"Error in parse_html_quiz: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при парсинге HTML теста: {str(e)}")

    if not questions:
        raise HTTPException(status_code=400, detail="В HTML не найдено ни одного вопроса. Убедитесь, что HTML содержит правильную структуру теста (например, div с классом 'que').")

    return {"ok": True, "questions": questions}

@app.post("/api/process-quiz/")
async def process_quiz(data: ProcessQuizRequest):
    questions = data.questions
    lecture_text = data.lecture_text

    if not lecture_text:
        raise HTTPException(status_code=400, detail="Поле 'lecture_text' отсутствует или пусто. Сначала загрузите PDF-лекцию и убедитесь, что она успешно обработалась.")

    if not questions:
        raise HTTPException(status_code=400, detail="Поле 'questions' отсутствует или пусто. Сначала распарсите HTML теста.")

    results = []

    for q in questions:
        qtext = q.get("question", "")
        qtype = detect_question_type(qtext)
        opts = q.get("options", [])
        is_short = q.get("is_short", False)

        if is_short or qtype == 'short':
            lec = lecture_text
            term_candidates = re.findall(r'([А-Яа-яЁёA-Za-z0-9 \-]{2,60})\s*[—\-:]\s*это\s+([^,.!?]+)', lec, re.IGNORECASE)
            term_candidates.extend(re.findall(r'([А-Яа-яЁёA-Za-z0-9 \-]{2,60})\s+является\s+([^,.!?]+)', lec, re.IGNORECASE))

            found = None
            question_lower = qtext.lower()
            for candidate_term, candidate_def in term_candidates:
                candidate_term_clean = candidate_term.strip().lower()
                candidate_def_clean = candidate_def.strip().lower()

                question_parts_to_match = [part.strip() for part in question_lower.split('это')[0].split('является')[0].split(',') if part.strip()]
                match_score = sum(1 for part in question_parts_to_match if part in candidate_def_clean)
                if match_score > 0:
                    found = {"answer": candidate_term_clean.title(), "excerpt": excerpt_around(lec, lec.find(candidate_term))}
                    break

            if not found:
                if 'электричество' in question_lower:
                    electro_match = re.search(r'([А-Яа-яЁёA-Za-z0-9 \-]+электричество[А-Яа-яЁёA-Za-z0-9 \-]*)\s*[—\-:]\s*([^,.!?]+)', lec, re.IGNORECASE)
                    if electro_match:
                        found = {"answer": electro_match.group(1).strip(), "excerpt": excerpt_around(lec, electro_match.start())}
                    else:
                        static_elec_match = re.search(r'(Статическое\s+электричество)\s*[—\-:]\s*([^,.!?]+)', lec, re.IGNORECASE)
                        if static_elec_match:
                            found = {"answer": static_elec_match.group(1).strip(), "excerpt": excerpt_around(lec, static_elec_match.start())}

            results.append({
                "question": qtext,
                "type": "short",
                "answer": found["answer"] if found else "",
                "excerpt": found["excerpt"] if found else "",
            })
            continue

        scored = []
        for opt in opts:
            score_result = score_option_by_lecture(lecture_text, opt, qtext)
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
            sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
            top = sorted_scores[0]
            second = sorted_scores[1] if len(sorted_scores) > 1 else None
            if second is None or (top["score"] - second["score"]) > max_score * 0.3:
                selected = [{"option": top["option"], "score": top["norm"], "snippets": top["snippets"]}]
            else:
                selected = [{"option": top["option"], "score": top["norm"], "snippets": top["snippets"]}]
        elif qtype == 'units':
            q_lower = qtext.lower()
            has_equiv = 'эквивалентн' in q_lower
            has_eff = 'эффективн' in q_lower
            relevant_opts = []
            for s in scored:
                opt_lower = s["option"].lower()
                if (has_equiv and ('зиверт' in opt_lower or 'зив' in opt_lower or 'sv' in opt_lower.lower())) or \
                   (has_eff and ('зиверт' in opt_lower or 'зив' in opt_lower or 'sv' in opt_lower.lower())):
                    relevant_opts.append(s)
                elif not (has_equiv or has_eff):
                    relevant_opts.append(s)

            if relevant_opts:
                candidates = [s for s in relevant_opts if s["norm"] >= 0.55]
                if not candidates:
                    fallback = [s for s in relevant_opts if s["norm"] >= 0.25]
                    selected = fallback
                else:
                    selected = candidates
            else:
                candidates = [s for s in scored if s["norm"] >= 0.55]
                if not candidates:
                    fallback = [s for s in scored if s["norm"] >= 0.25]
                    selected = fallback
                else:
                    selected = candidates

            selected.sort(key=lambda x: x["norm"], reverse=True)
        else:
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
