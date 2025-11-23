from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Лёгкая модель для Render Free (не превышает 512MB)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")

def cosine(a, b):
    return float(util.cos_sim(a, b)[0][0])

def answer_questions_from_html(lecture_text, html):
    soup = BeautifulSoup(html, "html.parser")

    questions = []

    # Ищем вопросы
    for block in soup.find_all(["div", "p", "li"]):
        txt = block.get_text(strip=True)
        if "?" in txt:
            questions.append({"question": txt, "options": []})

    # Ищем варианты ответов
    for opt in soup.find_all(["label", "li", "span"]):
        text = opt.get_text(strip=True)
        if len(text) > 3 and questions:
            questions[-1]["options"].append(text)

    result = ""
    for q in questions:
        question = q["question"]
        opts = q["options"]
        result += f"❓ {question}\n"

        # Открытый вопрос
        if not opts:
            sentences = [s for s in lecture_text.split("\n") if len(s) > 25]
            q_emb = model.encode(question)
            sent_emb = model.encode(sentences)
            sims = [cosine(q_emb, s) for s in sent_emb]
            best = sentences[int(np.argmax(sims))]
            result += f"✔ Ответ: {best}\n\n"
            continue

        # Вопрос с вариантами
        q_emb = model.encode(question)
        opt_emb = model.encode(opts)
        sims = [cosine(q_emb, o) for o in opt_emb]
        best = opts[int(np.argmax(sims))]
        result += f"✔ Ответ: {best}\n\n"

    return result
