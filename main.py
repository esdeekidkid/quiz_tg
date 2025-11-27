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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SESSION_STORAGE = {}

# --- Pydantic модели ---
class QuizHtmlRequest(BaseModel):
    html: str = Field(..., min_length=1)

class ProcessQuizRequest(BaseModel):
    questions: List[Dict[str, Any]]
    lecture_text: str = Field(..., min_length=1)

# --- Утилиты ---

def normalize_text(s):
    """Нормализация текста для сравнения"""
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip().lower()

def extract_full_sentence(text, position):
    """Извлекает полное предложение из текста по позиции"""
    if not text or position < 0:
        return ""
    
    # Ищем начало предложения (после точки, восклицательного или вопросительного знака)
    sentence_start = position
    for i in range(position - 1, -1, -1):
        if text[i] in '.!?\n':
            sentence_start = i + 1
            break
        elif i == 0:
            sentence_start = 0
    
    # Ищем конец предложения
    sentence_end = len(text)
    for i in range(position, len(text)):
        if text[i] in '.!?':
            sentence_end = i + 1
            break
    
    sentence = text[sentence_start:sentence_end].strip()
    
    # Если предложение слишком короткое, добавляем соседнее
    if len(sentence) < 100 and sentence_end < len(text):
        next_end = sentence_end
        for i in range(sentence_end, min(sentence_end + 500, len(text))):
            if text[i] in '.!?':
                next_end = i + 1
                break
        sentence = text[sentence_start:next_end].strip()
    
    return sentence

def find_text_similarity(text1, text2):
    """Вычисляет схожесть двух текстов (0-1)"""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def find_definition_by_text(lecture, definition_text):
    """Находит термин по его определению в лекции"""
    # Нормализуем входное определение
    def_normalized = normalize_text(definition_text)
    def_words = def_normalized.split()
    
    # Удаляем стоп-слова из определения
    stop_words = {'это', 'является', 'означает', 'представляет', 'собой', 'называется', 'ответ', 'вопрос'}
    def_words_filtered = [w for w in def_words if w not in stop_words and len(w) > 2]
    
    if len(def_words_filtered) < 3:
        return None
    
    # Ищем все определения в лекции с паттерном "ТЕРМИН - это ОПРЕДЕЛЕНИЕ"
    patterns = [
        r'([А-Яа-яЁё][А-Яа-яЁё\s\-]{2,60})\s*[—\-:]\s*это\s+([^.!?]{20,300}[.!?])',
        r'([А-Яа-яЁё][А-Яа-яЁё\s\-]{2,60})\s+[—\-]\s+([^.!?]{20,300}[.!?])',
        r'([А-Яа-яЁё][А-Яа-яЁё\s\-]{2,60})\s+является\s+([^.!?]{20,300}[.!?])',
    ]
    
    best_match = None
    best_score = 0.0
    
    for pattern in patterns:
        matches = re.finditer(pattern, lecture, re.IGNORECASE)
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Вычисляем схожесть определения из лекции с определением из вопроса
            similarity = find_text_similarity(definition, definition_text)
            
            # Также проверяем точное совпадение ключевых фраз
            def_lec_normalized = normalize_text(definition)
            key_phrase_matches = sum(1 for w in def_words_filtered[:10] if w in def_lec_normalized)
            phrase_ratio = key_phrase_matches / min(len(def_words_filtered), 10)
            
            # Комбинированный скор
            combined_score = (similarity * 0.5) + (phrase_ratio * 0.5)
            
            if combined_score > best_score and combined_score > 0.4:
                best_score = combined_score
                best_match = {
                    "term": term,
                    "definition": definition,
                    "position": match.start(),
                    "score": combined_score
                }
    
    return best_match

def score_option_by_lecture(lecture, option, question=""):
    """Оценивает опцию на основе лекции и контекста вопроса"""
    L = normalize_text(lecture)
    opt = normalize_text(option)
    q = normalize_text(question)

    score = 0
    snippets = []
    
    # Извлекаем ключевые контекстные слова из вопроса
    context_keywords = []
    
    # Для вопросов про единицы измерения
    if 'единиц' in q or 'измерения' in q:
        if 'эквивалентн' in q:
            context_keywords.append('эквивалентн')
        if 'эффективн' in q:
            context_keywords.append('эффективн')
        if 'доз' in q:
            context_keywords.append('доз')
    
    # Точное вхождение опции в лекцию
    exact_pattern = re.escape(opt)
    exact_matches = list(re.finditer(exact_pattern, L))
    exact_count = len(exact_matches)
    
    if exact_count > 0:
        base_score = 3 * (1 + exact_count)**0.5
        
        # Проверяем контекст вокруг найденного совпадения
        context_bonus = 0
        best_context_match = None
        
        for match in exact_matches:
            match_pos = match.start()
            # Берём контекст вокруг совпадения (500 символов)
            context_start = max(0, match_pos - 250)
            context_end = min(len(L), match_pos + 250)
            context = L[context_start:context_end]
            
            # Проверяем, есть ли контекстные слова рядом
            context_words_found = sum(1 for kw in context_keywords if kw in context)
            
            if context_words_found > context_bonus:
                context_bonus = context_words_found
                # Находим позицию в оригинальном тексте
                orig_pos = lecture.lower().find(opt, match_pos - 10)
                if orig_pos != -1:
                    best_context_match = extract_full_sentence(lecture, orig_pos)
        
        if context_bonus > 0:
            base_score *= (1 + context_bonus * 0.5)
            if best_context_match:
                snippets.append({
                    "why": f"exact_context (keywords: {context_bonus})",
                    "excerpt": best_context_match
                })
        else:
            # Если контекстные слова не найдены, снижаем балл для units
            if 'единиц' in q or 'измерения' in q:
                base_score *= 0.3
            if best_context_match:
                snippets.append({
                    "why": "exact",
                    "excerpt": best_context_match
                })
        
        score += base_score

    # Поиск определений
    def_patterns = [
        rf"{re.escape(opt)}\s*[—\-:]\s*это\s+([^.!?]+[.!?])",
        rf"{re.escape(opt)}\s+является\s+([^.!?]+[.!?])",
        rf"{re.escape(opt)}\s*[—\-]\s+([^.!?]+[.!?])",
    ]
    
    for pat in def_patterns:
        matches = list(re.finditer(pat, lecture, re.IGNORECASE))
        for match in matches:
            match_text = match.group(0)
            match_has_q_word = any(word in normalize_text(match_text) for word in q.split() if len(word) > 3)
            
            bonus = 3
            if match_has_q_word:
                bonus *= 1.5
            
            score += bonus
            full_sentence = extract_full_sentence(lecture, match.start())
            snippets.append({
                "why": f"definition{'_q_match' if match_has_q_word else ''}",
                "excerpt": full_sentence
            })

    # Пересечение слов
    opt_words = set(opt.split())
    if opt_words:
        matched_words = len(opt_words.intersection(set(L.split())))
        ratio = matched_words / len(opt_words)
        score += ratio * 2
        if ratio > 0:
            snippets.append({
                "why": "word-match",
                "matched": f"{matched_words}/{len(opt_words)}"
            })

    # Бонус за длинную опцию
    if len(opt) > 30 and exact_count > 0:
        score += 0.5

    return {"score": score, "snippets": snippets}

def detect_question_type(qtext):
    """Определяет тип вопроса"""
    q = normalize_text(qtext)
    
    # Проверка на короткий ответ
    if re.search(r'(какое слово пропущено|слово пропущено|впишите|введите|ответ\s*вопрос)', qtext, re.IGNORECASE):
        return 'short'
    
    # Единицы измерения
    if 'единиц' in q and 'измерения' in q:
        return 'units'
    
    # Single choice
    single_markers = ['какое из', 'какой из', 'как называется', 'что из', 'выберите один', 'что такое']
    for marker in single_markers:
        if marker in q:
            return 'single'

    # Multi choice
    multi_markers = ['какие', 'перечисл', 'классификация', 'входят в', 'относятся', 'назовите все', 'перечислите']
    for marker in multi_markers:
        if marker in q:
            return 'multi'

    return 'single'

def parse_html_quiz(html):
    """Парсит HTML теста"""
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
        raise HTTPException(status_code=400, detail="В HTML не найдено ни одного вопроса.")

    return {"ok": True, "questions": questions}

@app.post("/api/process-quiz/")
async def process_quiz(data: ProcessQuizRequest):
    questions = data.questions
    lecture_text = data.lecture_text

    if not lecture_text:
        raise HTTPException(status_code=400, detail="Поле 'lecture_text' отсутствует или пусто.")

    if not questions:
        raise HTTPException(status_code=400, detail="Поле 'questions' отсутствует или пусто.")

    results = []

    for q in questions:
        qtext = q.get("question", "")
        qtype = detect_question_type(qtext)
        opts = q.get("options", [])
        is_short = q.get("is_short", False)

        # Обработка вопросов с коротким ответом
        if is_short or qtype == 'short':
            # Ищем определение в вопросе
            match = find_definition_by_text(lecture_text, qtext)
            
            if match:
                results.append({
                    "question": qtext,
                    "type": "short",
                    "answer": match["term"],
                    "excerpt": extract_full_sentence(lecture_text, match["position"]),
                })
            else:
                results.append({
                    "question": qtext,
                    "type": "short",
                    "answer": "",
                    "excerpt": "Определение не найдено в лекции",
                })
            continue

        # Обработка вопросов с опциями
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
            # Выбираем один вариант с максимальным баллом
            sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
            if sorted_scores:
                top = sorted_scores[0]
                # Проверяем, есть ли явное превосходство
                if len(sorted_scores) > 1:
                    second = sorted_scores[1]
                    # Если разница большая, выбираем только топ
                    if top["score"] > second["score"] * 1.3:
                        selected = [{"option": top["option"], "score": top["norm"], "snippets": top["snippets"]}]
                    else:
                        selected = [{"option": top["option"], "score": top["norm"], "snippets": top["snippets"]}]
                else:
                    selected = [{"option": top["option"], "score": top["norm"], "snippets": top["snippets"]}]
                    
        elif qtype == 'units':
            # Для единиц измерения - строгий контекстный поиск
            q_lower = qtext.lower()
            relevant_opts = []
            
            # Проверяем каждую опцию на наличие контекстных слов рядом с ней
            for s in scored:
                # Если есть сниппет с exact_context - это наш кандидат
                has_context = any('exact_context' in snippet.get('why', '') for snippet in s['snippets'])
                if has_context:
                    relevant_opts.append(s)
            
            # Если нашли опции с контекстом, выбираем их
            if relevant_opts:
                relevant_opts.sort(key=lambda x: x["score"], reverse=True)
                selected = [{"option": s["option"], "score": s["norm"], "snippets": s["snippets"]} 
                           for s in relevant_opts]
            else:
                # Иначе берём топ по баллам
                candidates = [s for s in scored if s["norm"] >= 0.6]
                if candidates:
                    selected = [{"option": s["option"], "score": s["norm"], "snippets": s["snippets"]} 
                               for s in candidates]
                else:
                    # Fallback - топ-1
                    sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
                    if sorted_scores:
                        selected = [{"option": sorted_scores[0]["option"], 
                                   "score": sorted_scores[0]["norm"], 
                                   "snippets": sorted_scores[0]["snippets"]}]
                    
        else:  # multi
            candidates = [s for s in scored if s["norm"] >= 0.5]
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
            "selected": selected
        })

    return {"ok": True, "results": results}
