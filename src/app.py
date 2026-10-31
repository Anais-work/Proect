"""
Doc Assistant — REST API на FastAPI для RAG-системы (BERT + FAISS + GigaChat).

Запуск локально:
    uvicorn src.app:app --reload
Документация:
    http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.rag import build_index, rag_answer

app = FastAPI(title="Doc Assistant")

# Тяжёлые объекты (индекс FAISS, модель эмбеддингов BERT) грузим один раз при старте
index_state = None


@app.on_event("startup")
def load():
    global index_state
    index_state = build_index()  # строим/загружаем индекс один раз


class Query(BaseModel):
    question: str
    k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    if not 1 <= q.k <= 10:
        raise HTTPException(status_code=400, detail="k должно быть от 1 до 10")
    if index_state is None:
        raise HTTPException(status_code=503, detail="Индекс ещё не загружен")
    try:
        answer, sources = rag_answer(q.question, index_state, k=q.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {e}")
    return {"answer": answer, "sources": sources}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Простой веб-интерфейс для вопросов к сервису."""
    return INDEX_HTML


INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Business News RAG — Вопрос-Ответ</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
        body { background: #f4f6f9; color: #1e293b; margin: 0; padding: 30px 15px; }
        .container { max-width: 850px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 28px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        h1 { margin-top: 0; font-size: 24px; color: #0f172a; }
        .subtitle { color: #64748b; font-size: 14px; margin-bottom: 20px; }
        .input-group { display: flex; gap: 10px; margin-bottom: 12px; }
        input[type="text"] { flex: 1; padding: 12px 15px; font-size: 15px; border: 1px solid #cbd5e1; border-radius: 8px; outline: none; }
        input[type="text"]:focus { border-color: #2563eb; }
        button { background: #2563eb; color: #fff; border: none; padding: 12px 22px; font-size: 15px; font-weight: 600; border-radius: 8px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #94a3b8; cursor: not-allowed; }
        .examples { font-size: 13px; color: #475569; margin-bottom: 20px; }
        .examples span { display: inline-block; background: #e2e8f0; padding: 4px 10px; border-radius: 6px; margin: 3px 4px 3px 0; cursor: pointer; }
        .examples span:hover { background: #cbd5e1; }
        .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-top: 18px; display: none; }
        .answer-title { font-weight: 700; color: #0f172a; margin-bottom: 8px; font-size: 16px; }
        .answer-text { font-size: 16px; line-height: 1.55; color: #1e293b; background: #eff6ff; border-left: 4px solid #2563eb; padding: 14px; border-radius: 6px; }
        .meta { font-size: 13px; color: #64748b; margin-top: 12px; }
        .sources { margin-top: 16px; }
        .source-item { background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px; margin-top: 8px; font-size: 13px; }
        .source-header { font-weight: 600; color: #334155; margin-bottom: 4px; }
        .footer-links { margin-top: 22px; font-size: 13px; color: #64748b; text-align: right; }
        .footer-links a { color: #2563eb; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 RAG-система: Бизнес-новости (BERT + FAISS + GigaChat)</h1>
        <div class="subtitle">Задайте вопрос на английском или русском языке по базе бизнес-новостей</div>
        <div class="input-group">
            <input type="text" id="questionInput" placeholder="Введите ваш вопрос..." value="How many electric vehicle makers registered on the online portal for recycling lithium-ion battery waste, and which brands are mentioned?" />
            <button id="askBtn" onclick="sendQuestion()">Спросить</button>
        </div>
        <div class="examples">
            Примеры:
            <span onclick="setQ('What happened with the interest rates decided by the central bank?')">Процентные ставки ЦБ</span>
            <span onclick="setQ('Why did shares of Adani Power climb, and how much stake was acquired by GQG Partners?')">Акции Adani Power</span>
            <span onclick="setQ('Who won the UEFA Champions League football tournament in 2024?')">Проверка фильтра (Футбол)</span>
        </div>
        <div id="resultCard" class="card">
            <div class="answer-title">Ответ нейросети:</div>
            <div id="answerBox" class="answer-text"></div>
            <div id="tokensBox" class="meta"></div>
            <div id="sourcesBox" class="sources"></div>
        </div>
        <div class="footer-links">
            Техническая документация API: <a href="/docs" target="_blank">Swagger UI (/docs)</a>
        </div>
    </div>
    <script>
        function setQ(text) {
            document.getElementById('questionInput').value = text;
        }
        async function sendQuestion() {
            const input = document.getElementById('questionInput');
            const btn = document.getElementById('askBtn');
            const card = document.getElementById('resultCard');
            const answerBox = document.getElementById('answerBox');
            const tokensBox = document.getElementById('tokensBox');
            const sourcesBox = document.getElementById('sourcesBox');
            const q = input.value.trim();
            if (!q) return;

            btn.disabled = true;
            btn.textContent = 'Думаю...';
            card.style.display = 'block';
            answerBox.textContent = '⏳ Поиск в базе FAISS и генерация ответа через GigaChat...';
            tokensBox.textContent = '';
            sourcesBox.innerHTML = '';

            try {
                const resp = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q, k: 3 })
                });
                const data = await resp.json();
                answerBox.textContent = data.answer || 'Нет ответа';
                if (!resp.ok) {
                    answerBox.textContent = 'Ошибка ' + resp.status + ': ' + JSON.stringify(data.detail);
                    return;
                }
                if (data.sources && data.sources.length > 0 && !data.answer.toLowerCase().includes('not found') && !data.answer.toLowerCase().includes('не найдено')) {
                    let html = '<div class="answer-title" style="margin-top:14px;font-size:14px;">📚 Найденные источники в базе (FAISS):</div>';
                    data.sources.forEach((s, idx) => {
                        html += `<div class="source-item">
                            <div class="source-header">[${idx + 1}] Документ #${s.doc_id} | Категория: ${s.category} | L2-дистанция: ${s.distance.toFixed(4)}</div>
                            <div>${s.text}</div>
                        </div>`;
                    });
                    sourcesBox.innerHTML = html;
                }
            } catch (err) {
                answerBox.textContent = 'Ошибка соединения с сервером: ' + err;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Спросить';
            }
        }
    </script>
</body>
</html>"""
