from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

def answer_questions_from_html(lecture_text, html):
    soup = BeautifulSoup(html, "html.parser")

    questions = []
    for block in soup.find_all(["div", "p", "li"]):
        txt = block.get_text(strip=True)
        if "?" in txt:
            questions.append({"question": txt, "options": []})

    for opt in soup.find_all(["label", "li", "span"]):
        text = opt.get_text(strip=True)
        if len(text) > 3 and questions:
            questions[-1]["options"].append(text)

    result = ""
    lecture_sentences = [s.strip() for s in lecture_text.split("\n") if len(s.strip()) > 25]

    for q in questions:
        question = q["question"]
        opts = q["options"]
        result += f"❓ {question}\n"

        if not opts:
            best_match = process.extractOne(query=question, choices=lecture_sentences, scorer=fuzz.token_sort_ratio)
            if best_match:
                result += f"✔ Ответ: {best_match[0]}\n\n"
            continue

        best_opt = process.extractOne(query=question, choices=opts, scorer=fuzz.token_sort_ratio)
        if best_opt:
            result += f"✔ Ответ: {best_opt[0]}\n\n"

    return result
