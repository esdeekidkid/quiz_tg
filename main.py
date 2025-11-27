# --- Новый маршрут для парсинга HTML ---
@app.post("/api/parse-quiz-html/")
async def parse_quiz_html( dict): # Принимаем JSON из тела запроса
    html = data.get("html", "")
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