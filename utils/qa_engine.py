from bs4 import BeautifulSoup
from rapidfuzz import process

def answer_questions_from_html(pdf_text, html):
    soup = BeautifulSoup(html, "html.parser")
    questions = soup.find_all(["p", "li"])

    pdf_lines = pdf_text.split("\n")

    out = []

    for q in questions:
        qtext = q.get_text(strip=True)
        if not qtext:
            continue
        
        match, score, *_ = process.extractOne(qtext, pdf_lines)
        out.append(f"Вопрос: {qtext}\nОтвет: {match}\n")

    return "\n".join(out)
