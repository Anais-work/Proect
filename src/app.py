"""
Doc Assistant — REST API на FastAPI для RAG-системы (BERT + FAISS + GigaChat).

Запуск локально:
    uvicorn src.app:app --reload
Документация:
    http://127.0.0.1:8000/docs
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agents import run_framework_agent, run_json_agent, run_native_agent, tool_calc
from src.classifier import get_classifier
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


class ClassifyRequest(BaseModel):
    text: str


class CalcRequest(BaseModel):
    expression: str


class AgentRequest(BaseModel):
    question: str
    mode: Literal["native", "json", "framework"] = "native"
    max_iterations: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/calc")
def calculate(req: CalcRequest):
    if not req.expression.strip():
        raise HTTPException(status_code=400, detail="Пустое арифметическое выражение")
    try:
        result = tool_calc(req.expression)
        return {"expression": req.expression, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка вычисления: {e}")


@app.post("/classify")
def classify(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Пустой текст")
    try:
        clf = get_classifier()
        return clf.predict(req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {e}")


@app.post("/agent")
def agent_endpoint(req: AgentRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    if not 1 <= req.max_iterations <= 15:
        raise HTTPException(status_code=400, detail="max_iterations должно быть от 1 до 15")
    try:
        if req.mode == "native":
            return run_native_agent(req.question, max_iterations=req.max_iterations, trace=False)
        if req.mode == "json":
            return run_json_agent(req.question, max_steps=req.max_iterations, trace=False)
        return run_framework_agent(req.question, max_iterations=req.max_iterations, trace=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка агента: {e}")


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
    <title>Doc Assistant — RAG, Калькулятор, Классификатор и ИИ-Агенты</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
        body { background: #f4f6f9; color: #1e293b; margin: 0; padding: 30px 15px; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 28px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        h1 { margin-top: 0; font-size: 24px; color: #0f172a; }
        .subtitle { color: #64748b; font-size: 14px; margin-bottom: 18px; }
        .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; border-bottom: 1px solid #e2e8f0; padding-bottom: 12px; }
        .tab-btn { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; padding: 8px 14px; font-size: 14px; font-weight: 600; border-radius: 8px; cursor: pointer; }
        .tab-btn.active { background: #2563eb; color: #fff; border-color: #2563eb; }
        .tab-btn:hover:not(.active) { background: #e2e8f0; }
        .panel { display: none; }
        .panel.active { display: block; }
        .input-group { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
        input[type="text"], select, textarea { flex: 1; padding: 12px 15px; font-size: 15px; border: 1px solid #cbd5e1; border-radius: 8px; outline: none; background: #fff; }
        select { flex: 0 0 auto; min-width: 170px; }
        textarea { min-height: 90px; resize: vertical; }
        input[type="text"]:focus, select:focus, textarea:focus { border-color: #2563eb; }
        button.action-btn { background: #2563eb; color: #fff; border: none; padding: 12px 22px; font-size: 15px; font-weight: 600; border-radius: 8px; cursor: pointer; }
        button.action-btn:hover { background: #1d4ed8; }
        button.action-btn:disabled { background: #94a3b8; cursor: not-allowed; }
        .examples { font-size: 13px; color: #475569; margin-bottom: 16px; }
        .examples span { display: inline-block; background: #e2e8f0; padding: 4px 10px; border-radius: 6px; margin: 3px 4px 3px 0; cursor: pointer; }
        .examples span:hover { background: #cbd5e1; }
        .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-top: 18px; display: none; }
        .answer-title { font-weight: 700; color: #0f172a; margin-bottom: 8px; font-size: 16px; }
        .answer-text { font-size: 16px; line-height: 1.55; color: #1e293b; background: #eff6ff; border-left: 4px solid #2563eb; padding: 14px; border-radius: 6px; white-space: pre-wrap; }
        .meta { font-size: 13px; color: #64748b; margin-top: 10px; }
        .sources, .steps { margin-top: 14px; }
        .source-item, .step-item { background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px; margin-top: 8px; font-size: 13px; }
        .source-header, .step-header { font-weight: 600; color: #334155; margin-bottom: 4px; }
        .code-line { font-family: monospace; background: #f1f5f9; padding: 4px 6px; border-radius: 4px; margin-top: 4px; word-break: break-word; }
        .footer-links { margin-top: 22px; font-size: 13px; color: #64748b; text-align: right; }
        .footer-links a { color: #2563eb; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Doc Assistant: ИИ-Агент, Калькулятор, RAG и Классификатор</h1>
        <div class="subtitle">Бизнес-ассистент на базе PyTorch, BERT + FAISS, GigaChat и LangGraph</div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('agent', this)">🧠 ИИ-Агент (3 инструмента)</button>
            <button class="tab-btn" onclick="switchTab('calc', this)">🧮 Калькулятор</button>
            <button class="tab-btn" onclick="switchTab('rag', this)">📚 RAG Поиск</button>
            <button class="tab-btn" onclick="switchTab('classify', this)">🏷️ Классификатор</button>
        </div>

        <!-- Вкладка 1: ИИ-Агент -->
        <div id="panel-agent" class="panel active">
            <div class="input-group">
                <input type="text" id="agentInput" value="Найди в базе новостей информацию о ставке центрального банка, определи тему найденного текста и посчитай 125 * 4 + 50." placeholder="Составной запрос к агенту..." />
                <select id="agentMode">
                    <option value="native">Native (tool_calls)</option>
                    <option value="json">JSON (SYSTEM + regex)</option>
                    <option value="framework">Framework (LangGraph)</option>
                </select>
                <button id="agentBtn" class="action-btn" onclick="sendAgent()">Запустить</button>
            </div>
            <div class="examples">
                Примеры:
                <span onclick="document.getElementById('agentInput').value='Найди в базе новостей информацию о ставке центрального банка, определи тему найденного текста и посчитай 125 * 4 + 50.'">Составной запрос (поиск + тема + расчёт)</span>
                <span onclick="document.getElementById('agentInput').value='Посчитай (2450 - 450) / 8 + 15 * 3'">Только калькулятор</span>
                <span onclick="document.getElementById('agentInput').value='Найди новости про Adani Power и GQG Partners'">Только поиск</span>
            </div>
            <div id="agentCard" class="card">
                <div class="answer-title">Финальный ответ агента:</div>
                <div id="agentAnswer" class="answer-text"></div>
                <div id="agentMeta" class="meta"></div>
                <div id="agentSteps" class="steps"></div>
            </div>
        </div>

        <!-- Вкладка 2: Калькулятор -->
        <div id="panel-calc" class="panel">
            <div class="input-group">
                <input type="text" id="calcInput" value="125 * 4 + 50" placeholder="Введите выражение, например 2*(3+4)" />
                <button id="calcBtn" class="action-btn" onclick="sendCalc()">Вычислить</button>
            </div>
            <div class="examples">
                Примеры:
                <span onclick="document.getElementById('calcInput').value='125 * 4 + 50'">125 * 4 + 50</span>
                <span onclick="document.getElementById('calcInput').value='2 * (3 + 4)'">2 * (3 + 4)</span>
                <span onclick="document.getElementById('calcInput').value='(1500 - 300) / 8'">(1500 - 300) / 8</span>
                <span onclick="document.getElementById('calcInput').value='2+'">Некорректное (2+)</span>
            </div>
            <div id="calcCard" class="card">
                <div class="answer-title">Результат tool_calc:</div>
                <div id="calcAnswer" class="answer-text"></div>
            </div>
        </div>

        <!-- Вкладка 3: RAG -->
        <div id="panel-rag" class="panel">
            <div class="input-group">
                <input type="text" id="questionInput" placeholder="Введите ваш вопрос..." value="How many electric vehicle makers registered on the online portal for recycling lithium-ion battery waste, and which brands are mentioned?" />
                <button id="askBtn" class="action-btn" onclick="sendQuestion()">Спросить</button>
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
        </div>

        <!-- Вкладка 4: Классификатор -->
        <div id="panel-classify" class="panel">
            <div class="input-group">
                <textarea id="classifyInput" placeholder="Введите новостной текст на английском...">Sensex and Nifty surged as banking stocks rallied on strong quarterly earnings.</textarea>
                <button id="classifyBtn" class="action-btn" onclick="sendClassify()">Определить тему</button>
            </div>
            <div class="examples">
                Примеры:
                <span onclick="document.getElementById('classifyInput').value='Reserve Bank kept repo rate unchanged amid inflation concerns.'">Банки / Ставка</span>
                <span onclick="document.getElementById('classifyInput').value='Tech startup launched a new artificial intelligence cloud software platform.'">Технологии</span>
            </div>
            <div id="classifyCard" class="card">
                <div class="answer-title">Предсказанная категория:</div>
                <div id="classifyAnswer" class="answer-text"></div>
                <div id="classifyProbs" class="sources"></div>
            </div>
        </div>

        <div class="footer-links">
            Техническая документация API: <a href="/docs" target="_blank">Swagger UI (/docs)</a>
        </div>
    </div>
    <script>
        function escapeHtml(str) {
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }
        function switchTab(name, btn) {
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('panel-' + name).classList.add('active');
            btn.classList.add('active');
        }
        function setQ(text) {
            document.getElementById('questionInput').value = text;
        }
        async function sendAgent() {
            const q = document.getElementById('agentInput').value.trim();
            const mode = document.getElementById('agentMode').value;
            const btn = document.getElementById('agentBtn');
            const card = document.getElementById('agentCard');
            const ansBox = document.getElementById('agentAnswer');
            const metaBox = document.getElementById('agentMeta');
            const stepsBox = document.getElementById('agentSteps');
            if (!q) return;

            btn.disabled = true;
            btn.textContent = 'Выполняю...';
            card.style.display = 'block';
            ansBox.textContent = '⏳ Запуск агента (' + mode + ')...';
            metaBox.textContent = '';
            stepsBox.innerHTML = '';

            try {
                const resp = await fetch('/agent', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q, mode: mode, max_iterations: 6 })
                });
                const data = await resp.json();
                if (!resp.ok) {
                    ansBox.textContent = 'Ошибка ' + resp.status + ': ' + JSON.stringify(data.detail);
                    return;
                }
                ansBox.textContent = data.answer || 'Нет ответа';
                metaBox.textContent = `Режим: ${data.mode} | Итераций: ${data.iterations} | Вызовов инструментов: ${(data.steps || []).length}`;
                if (data.steps && data.steps.length > 0) {
                    let html = '<div class="answer-title" style="margin-top:12px;font-size:14px;">🔍 Лог шагов (трассировка):</div>';
                    data.steps.forEach(s => {
                        const toolName = escapeHtml(s.tool || 'parse_error');
                        const argsText = escapeHtml(typeof s.args === 'object' ? JSON.stringify(s.args) : (s.args ?? ''));
                        const resText = escapeHtml(s.result || s.error || '');
                        html += `<div class="step-item">
                            <div class="step-header">[step ${s.step}] tool=${toolName}</div>
                            <div class="code-line">args: ${argsText}</div>
                            <div class="code-line">result: ${resText}</div>
                        </div>`;
                    });
                    stepsBox.innerHTML = html;
                }
            } catch (err) {
                ansBox.textContent = 'Ошибка соединения с сервером: ' + err;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Запустить';
            }
        }
        async function sendCalc() {
            const expr = document.getElementById('calcInput').value.trim();
            const btn = document.getElementById('calcBtn');
            const card = document.getElementById('calcCard');
            const ansBox = document.getElementById('calcAnswer');
            if (!expr) return;
            btn.disabled = true;
            card.style.display = 'block';
            ansBox.textContent = '⏳ Вычисление...';
            try {
                const resp = await fetch('/calc', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ expression: expr })
                });
                const data = await resp.json();
                ansBox.textContent = resp.ok ? `${data.expression} = ${data.result}` : `Ошибка ${resp.status}: ${data.detail}`;
            } catch (err) {
                ansBox.textContent = 'Ошибка соединения с сервером: ' + err;
            } finally {
                btn.disabled = false;
            }
        }
        async function sendClassify() {
            const text = document.getElementById('classifyInput').value.trim();
            const btn = document.getElementById('classifyBtn');
            const card = document.getElementById('classifyCard');
            const ansBox = document.getElementById('classifyAnswer');
            const probsBox = document.getElementById('classifyProbs');
            if (!text) return;
            btn.disabled = true;
            card.style.display = 'block';
            ansBox.textContent = '⏳ Классификация...';
            probsBox.innerHTML = '';
            try {
                const resp = await fetch('/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await resp.json();
                if (!resp.ok) {
                    ansBox.textContent = 'Ошибка ' + resp.status + ': ' + data.detail;
                    return;
                }
                ansBox.textContent = `${data.category} (уверенность: ${(data.confidence * 100).toFixed(2)}%)`;
                if (data.probabilities) {
                    const rows = Object.entries(data.probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([k, v]) => `<div class="source-item"><b>${escapeHtml(k)}</b>: ${(v * 100).toFixed(2)}%</div>`)
                        .join('');
                    probsBox.innerHTML = rows;
                }
            } catch (err) {
                ansBox.textContent = 'Ошибка соединения с сервером: ' + err;
            } finally {
                btn.disabled = false;
            }
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
                            <div class="source-header">[${idx + 1}] Документ #${s.doc_id} | Категория: ${escapeHtml(s.category)} | L2-дистанция: ${s.distance.toFixed(4)}</div>
                            <div>${escapeHtml(s.text)}</div>
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
