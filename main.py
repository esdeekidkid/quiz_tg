# main.py — Quiz helper (improved B + V)
# Python 3.11+ recommended (but should work on 3.10/3.12)
#
# Features:
# - Accept PDF upload, extract text (PyMuPDF / fitz)
# - When a new PDF is uploaded, clear previous sessions (so only latest PDF used)
# - Parse HTML with BeautifulSoup to get questions and answers (robust selectors)
# - Check answers with combined heuristics:
#     * exact substring match
#     * fuzzy match (rapidfuzz if available)
#     * keyword overlap between question and lecture
#     * context snippet scoring
# - For open (short-answer) questions returns best matching snippet + confidence
# - Endpoints:
#     POST /upload_pdf       -> upload PDF, returns session_id
#     POST /process_quiz     -> parse HTML -> list questions (include session_hint)
#     POST /check_answers    -> given question+answers+session -> analysis+correct_answers
#     POST /reset_session/{session_id}
#     GET  /clear_all
#
# Memory: only stores last uploaded lecture (by design), minimal footprint.
# No Supabase, no external storage.

from __future__ import annotations
import re
import time
import logging
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup

# try to import rapidfuzz for fuzzy matching; optional
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quiz_helper")

app = FastAPI(title="Quiz Helper (improved B+V)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store
# We will keep only one "active" lecture (to follow your requirement:
# when new PDF uploaded all previous data is forgotten).
# But we return a session_id so front can verify.
ACTIVE_SESSION: Optional[Dict[str, Any]] = None

def make_session_id() -> str:
    return str(int(time.time() * 1000))

# Models
class HtmlPayload(BaseModel):
    html: str

class CheckPayload(BaseModel):
    question: str
    answers: List[str] = []
    session_id: str

# Utilities
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for p in doc:
        try:
            t = p.get_text("text")
            if t:
                parts.append(t)
        except Exception:
            # fallback: ignore bad page
            continue
    return "\n".join(parts)

def simple_normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\r', ' ', s)
    s = re.sub(r'\n+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    # keep letters, numbers, spaces
    s = re.sub(r'[^0-9a-zа-яё\s]', '', s)
    return s.strip()

def fuzzy_score(a: str, b: str) -> float:
    """Return similarity in [0,1]. Use rapidfuzz if available for quality."""
    if not a or not b:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return fuzz.partial_ratio(a, b) / 100.0
    # fallback heuristic: overlap of word sets
    aw = set(a.split())
    bw = set(b.split())
    if not aw or not bw:
        return 0.0
    inter = len(aw & bw)
    denom = max(len(aw), len(bw))
    return inter / denom

def split_sentences(text: str) -> List[str]:
    # naive split, OK for our purposes
    parts = re.split(r'(?<=[\.\?\!\n])\s+', text)
    sents = [p.strip() for p in parts if p.strip()]
    return sents

def context_snippet(text: str, term: str, ctx: int = 100) -> str:
    """Return snippet around first occurrence of term (or a word from term)."""
    tnorm = text
    idx = tnorm.find(term)
    if idx == -1:
        # try words
        for w in term.split():
            if len(w) < 3:
                continue
            idx = tnorm.find(w)
            if idx != -1:
                break
    if idx == -1:
        # no location
        return ""
    start = max(0, idx - ctx)
    end = min(len(tnorm), idx + len(term) + ctx)
    return tnorm[start:end].strip()

# Heuristic scoring improved:
# - exact substring match -> strong
# - fuzzy matching against best sentence window
# - keyword overlap with question increases score slightly
# - small penalty if answer appears many times (ambiguous)
def score_answer_against_lecture(ans: str, question: str, lecture_norm: str) -> Dict[str, Any]:
    a_norm = simple_normalize(ans)
    q_norm = simple_normalize(question)
    result = {"answer": ans, "score": 0.0, "found": None, "notes": []}

    if not a_norm:
        result["notes"].append("empty_answer")
        return result

    # 1) exact substring
    if a_norm in lecture_norm:
        result["score"] += 0.55
        result["found"] = context_snippet(lecture_norm, a_norm, ctx=120)
        result["notes"].append("exact_substring")

    # 2) fuzzy search for best matching sentence/window
    # search through sentences for best match
    best_ratio = 0.0
    best_piece = None
    for piece in split_sentences(lecture_norm):
        if len(piece) < 20:
            continue
        r = fuzzy_score(a_norm, piece)
        if r > best_ratio:
            best_ratio = r
            best_piece = piece
    if best_ratio > 0.5:
        # if already had exact_substring, combine; else give notable score
        bonus = 0.45 * best_ratio
        result["score"] += bonus
        if not result["found"]:
            result["found"] = best_piece
        result["notes"].append(f"fuzzy_match:{round(best_ratio,3)}")

    # 3) question-word overlap
    q_words = [w for w in re.findall(r'[а-яёa-z0-9]{3,}', q_norm)]
    if q_words:
        awords = set(a_norm.split())
        overlap = sum(1 for w in q_words if w in awords)
        if overlap:
            # small boost
            boost = 0.08 * (overlap / max(1, len(q_words)))
            result["score"] += boost
            result["notes"].append(f"q_overlap:{overlap}")

    # 4) sanity caps
    if result["score"] > 0.999:
        result["score"] = 1.0

    # If nothing found but answer words appear across lecture, give minimal score
    if result["score"] <= 0 and a_norm:
        # check token overlap anywhere
        tokens = a_norm.split()
        hits = sum(1 for t in tokens if t and (' ' + t + ' ') in (' ' + lecture_norm + ' '))
        if hits:
            result["score"] += 0.15 * (hits / max(1, len(tokens)))
            result["notes"].append(f"token_hits:{hits}")

    # final round: normalize to [0,1]
    result["score"] = max(0.0, min(1.0, round(result["score"], 3)))
    if not result["found"]:
        result["found"] = "—"
    return result

# Better selection strategy for correct answers:
# - pick answers with score >= threshold (absolute) and also consider gaps between top
# - if multi-select is likely (more than 1 answer text looks present), include multiple
# - if only one answer clearly best (gap > 0.2) return single
def pick_correct_answers(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    # sort descending
    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    top_score = results_sorted[0]["score"]
    # threshold absolute
    threshold = 0.35
    # choose candidates >= threshold
    candidates = [r for r in results_sorted if r["score"] >= threshold]
    if not candidates:
        # fallback: return top 1 if > 0.2
        if top_score >= 0.2:
            return [results_sorted[0]]
        return []
    # if many have same top_score (within small epsilon) return them
    eps = 0.05
    top_candidates = [r for r in candidates if top_score - r["score"] <= eps]
    # If top candidate is clearly ahead of next (gap>0.2) return only top
    if len(results_sorted) > 1:
        second_score = results_sorted[1]["score"]
        if top_score - second_score > 0.2 and top_score >= threshold:
            return [results_sorted[0]]
    # Otherwise return top_candidates (could be multiple)
    return top_candidates

# Endpoints

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF. When a new PDF is uploaded we clear any previous session (as requested).
    Returns session_id and small snippet.
    """
    global ACTIVE_SESSION
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    content = await file.read()
    text = extract_text_from_pdf_bytes(content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    # Clear prior data (user requested behavior)
    ACTIVE_SESSION = None

    sid = make_session_id()
    ACTIVE_SESSION = {
        "session_id": sid,
        "text": text,
        "normalized_text": simple_normalize(text),
        "uploaded_at": time.time()
    }
    logger.info(f"New PDF uploaded, session={sid}, len_text={len(text)}")
    # Return short snippet (first 1000 characters)
    snippet = text[:1000] + ("..." if len(text) > 1000 else "")
    return {"session_id": sid, "message": "PDF uploaded and active", "snippet": snippet}

@app.post("/process_quiz")
def process_quiz(payload: HtmlPayload, session_id: Optional[str] = Query(None, description="optional session hint")):
    """
    Parse HTML and return questions with answers.
    If session_id provided, we just echo it as hint for frontend to use check_answers.
    """
    soup = BeautifulSoup(payload.html, "html.parser")
    q_blocks = soup.select(".que")
    questions = []

    # If no .que found, try to heuristically find question blocks (e.g., <fieldset> with legend)
    if not q_blocks:
        q_blocks = soup.select("fieldset") or soup.select("form") or [soup]

    for qb in q_blocks:
        # try several places for the question text
        qtext_block = qb.select_one(".qtext") or qb.select_one(".formulation") or qb.select_one("legend") or qb
        qtext = qtext_block.get_text(" ", strip=True) if qtext_block else ""
        # gather answers with different strategies
        answers = []
        # typical Moodle structure: .answer .d-flex p
        for p in qb.select(".answer .d-flex p"):
            t = p.get_text(" ", strip=True)
            if t:
                answers.append(t)
        # fallback: labels next to inputs
        if not answers:
            for label in qb.select("label"):
                t = label.get_text(" ", strip=True)
                if t:
                    answers.append(t)
        # fallback: li elements
        if not answers:
            for li in qb.select("li"):
                t = li.get_text(" ", strip=True)
                if t:
                    answers.append(t)
        # fallback: any input adjacent texts
        if not answers:
            # look for immediate paragraphs inside block
            for p in qb.select("p"):
                t = p.get_text(" ", strip=True)
                if t and len(t) < 280:
                    answers.append(t)

        # dedupe preserving order
        seen = set()
        clean_answers = []
        for a in answers:
            if not a:
                continue
            if a in seen:
                continue
            seen.add(a)
            clean_answers.append(a)

        # If it's a short-answer question (input text present), we keep answers empty and mark type
        is_short = bool(qb.select_one("input[type='text'], input[type='search'], textarea"))

        if not qtext.strip():
            continue
        questions.append({
            "question": qtext,
            "answers": clean_answers,
            "is_short": is_short
        })

    return {"questions": questions, "session_hint": session_id or (ACTIVE_SESSION["session_id"] if ACTIVE_SESSION else None)}

@app.post("/check_answers")
def check_answers(payload: CheckPayload):
    """
    For given question+answers+session_id compute analysis and return guessed correct answers.
    For short (open) questions, returns snippet with confidence.
    """
    if not ACTIVE_SESSION:
        raise HTTPException(status_code=404, detail="No active PDF session. Upload PDF first.")

    # validate session_id matches active or ignore (we keep only one active)
    if payload.session_id != ACTIVE_SESSION["session_id"]:
        # if session mismatch, refuse; this helps frontend avoid mistakes
        raise HTTPException(status_code=400, detail="Session_id mismatch. Upload PDF and use returned session_id.")

    lecture_norm = ACTIVE_SESSION["normalized_text"]
    question = payload.question
    answers = payload.answers or []

    # If it's an open question (no answers provided), try to find best sentence / phrase
    if not answers:
        # try to locate best matching fragment for question using fuzzy over sentences
        q_norm = simple_normalize(question)
        best_score = 0.0
        best_frag = None
        for sent in split_sentences(lecture_norm):
            if len(sent) < 20:
                continue
            r = fuzzy_score(q_norm, sent)
            if r > best_score:
                best_score = r
                best_frag = sent
        confidence = round(min(1.0, best_score), 3)
        # try exact phrase: if question contains phrase like "это ____" maybe answer is next word in original text:
        # But safer: return best_frag as snippet
        return {"open_answer": best_frag or "—", "confidence": confidence}

    # For multi-choice, score each answer
    results = []
    for a in answers:
        res = score_answer_against_lecture(a, question, lecture_norm)
        results.append(res)

    correct = pick_correct_answers(results)
    debug = {"counts": len(results)}
    return {"analysis": results, "correct_answers": correct, "debug": debug}

@app.post("/reset_session/{session_id}")
def reset_session(session_id: str):
    global ACTIVE_SESSION
    if ACTIVE_SESSION and ACTIVE_SESSION["session_id"] == session_id:
        ACTIVE_SESSION = None
        return {"ok": True}
    return {"ok": False, "reason": "session_mismatch_or_not_found"}

@app.get("/clear_all")
def clear_all():
    global ACTIVE_SESSION
    ACTIVE_SESSION = None
    return {"ok": True}