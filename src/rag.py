
import os
import sys
import re
import json
import uuid
import base64
import argparse
import warnings
import urllib3
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Подавление предупреждений SSL и HuggingFace
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# Настройка путей
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
data_dir = os.path.join(project_root, 'data', 'processed')
load_dotenv(os.path.join(project_root, '.env'))


class BusinessRAG:
    """
    RAG-система по базе данных бизнес-новостей.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device_name: str = None):
        self.device = torch.device(device_name if device_name else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"🔧 Загрузка модели BERT ({model_name}) на {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.bert_model.eval()

        self.index = None
        self.chunks_metadata = []
        self.giga_token = None

    def create_chunks(self, df: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        ШАГ 2 из RAG.pdf:
        Разбиение текстов на чанки длиной 500 символов с перекрытием 50 символов.
        """
        step = chunk_size - chunk_overlap
        all_chunks = []

        print(f"✂️ Выполняем чанкинг (размер={chunk_size}, перекрытие={chunk_overlap})...")
        for doc_id, row in df.iterrows():
            text = str(row['text']).strip()
            category = row.get('category', 'unknown')

            if len(text) <= chunk_size:
                all_chunks.append({
                    'chunk_id': len(all_chunks),
                    'doc_id': doc_id,
                    'category': category,
                    'text': text
                })
                continue

            for start in range(0, len(text), step):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                if len(chunk_text) >= 40:
                    all_chunks.append({
                        'chunk_id': len(all_chunks),
                        'doc_id': doc_id,
                        'category': category,
                        'text': chunk_text
                    })

        print(f"✅ Сформировано {len(all_chunks)} чанков из {len(df)} документов.")
        return all_chunks

    def embed_texts(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """
        ШАГ 1 & 2a из RAG.pdf:
        Преобразование текста в эмбеддинги BERT через Mean Pooling и L2-нормализацию.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                all_embeddings.append(normed.cpu().numpy().astype('float32'))

        return np.vstack(all_embeddings)

    def load_or_build_index(self, train_csv_path: str = None, force_rebuild: bool = False):
        """
        ШАГ 3 из RAG.pdf:
        Загрузка или построение векторного индекса FAISS IndexFlatL2.
        """
        index_cache_path = os.path.join(data_dir, 'faiss_rag_index.bin')
        meta_cache_path = os.path.join(data_dir, 'faiss_chunks_meta.json')

        if not force_rebuild and os.path.exists(index_cache_path) and os.path.exists(meta_cache_path):
            print(f"⚡ Загрузка закэшированного FAISS индекса из: {index_cache_path}")
            self.index = faiss.read_index(index_cache_path)
            with open(meta_cache_path, 'r', encoding='utf-8') as f:
                self.chunks_metadata = json.load(f)
            print(f"✅ Векторный индекс готов! Всего векторов в БД: {self.index.ntotal}")
            return

        if not train_csv_path:
            train_csv_path = os.path.join(data_dir, 'train_data.csv')
            if not os.path.exists(train_csv_path):
                train_csv_path = os.path.join(data_dir, 'business_data_cleaned.csv')

        print(f"📥 Чтение обучающей выборки отличного качества: {train_csv_path}")
        df = pd.read_csv(train_csv_path)

        self.chunks_metadata = self.create_chunks(df, chunk_size=500, chunk_overlap=50)
        chunk_texts = [c['text'] for c in self.chunks_metadata]

        print(f"🧠 Расчет BERT-эмбеддингов для {len(chunk_texts)} чанков...")
        embeddings = self.embed_texts(chunk_texts, batch_size=64)

        dim = embeddings.shape[1]
        print(f"📦 Инициализация FAISS faiss.IndexFlatL2 (размерность {dim})...")
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, index_cache_path)
        with open(meta_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)

        print(f"💾 Индекс сохранен в {index_cache_path}. Всего элементов: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        ШАГ 4 из RAG.pdf:
        ANN-поиск top-k (от 5 до 10) релевантных чанков по запросу.
        """
        if self.index is None:
            raise RuntimeError("Индекс FAISS не инициализирован. Вызовите load_or_build_index().")

        k = max(5, min(10, top_k))
        q_emb = self.embed_texts([query])
        distances, indices = self.index.search(q_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            item = self.chunks_metadata[idx].copy()
            item['distance'] = float(dist)
            results.append(item)

        return results

    def get_gigachat_token(self) -> str:
        """Получение access token для GigaChat API."""
        if self.giga_token:
            return self.giga_token

        client_id = os.getenv("GIGA_CLIENT_ID")
        client_secret = os.getenv("GIGA_CLIENT_SECRET")
        if not client_id or not client_secret:
            return None

        auth_b64 = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_b64}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded"
        }
        try:
            resp = requests.post(
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                headers=headers,
                data={"scope": os.getenv("GIGA_SCOPE", "GIGACHAT_API_PERS")},
                verify=False,
                timeout=10
            )
            if resp.status_code == 200:
                self.giga_token = resp.json().get("access_token")
                return self.giga_token
        except Exception:
            pass
        return None

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        ШАГ 5 из RAG.pdf:
        Сборка контекста + строгий промпт против галлюцинаций + вызов LLM + источники.
        Поддерживает вопросы на английском и русском языках.
        """
        retrieved_chunks = self.search(question, top_k=top_k)

        # Проверка порога релевантности (L2 distance):
        # Если ближайший чанк имеет дистанцию > 0.85 (например, вопрос 'Как дела' с L2=1.20+),
        # значит в базе нет подходящих текстов вообще.
        is_russian = any('\u0400' <= char <= '\u04FF' for char in question)
        if not retrieved_chunks or retrieved_chunks[0]['distance'] > 0.85:
            not_found_msg = (
                "в документах не найдено (вопрос не относится к тематике базы знаний)."
                if is_russian else
                "Not found in documents (query is unrelated to the knowledge base)."
            )
            return {
                "question": question,
                "answer": not_found_msg,
                "sources": [],
                "top_k": len(retrieved_chunks),
                "usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            }

        # Склеиваем фрагменты
        context_parts = []
        for i, ch in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[{i}] {ch['text']}")
        context_str = "\n\n".join(context_parts)

        if is_russian:
            prompt = (
                "Ответь на вопрос ТОЛЬКО по приведённым фрагментам. "
                "Если в фрагментах нет информации для ответа — ответь строго 'в документах не найдено'.\n\n"
                f"Фрагменты:\n{context_str}\n\n"
                f"Вопрос: {question}"
            )
            system_role = "Ты строгий финансовый аналитик. Отвечай строго по фактам из предоставленных фрагментов."
        else:
            prompt = (
                "You are provided with several document excerpts below. "
                "Answer the user question using the facts from these excerpts accurately and concisely. "
                "If the excerpts do not contain relevant information to answer the question, respond strictly with 'Not found in documents'.\n\n"
                f"Excerpts:\n{context_str}\n\n"
                f"Question: {question}"
            )
            system_role = "You are an expert business research assistant. Answer based on the excerpts."

        token = self.get_gigachat_token()
        answer = "Не удалось подключиться к GigaChat API (проверьте ключи в .env)."
        usage = {}

        if token:
            chat_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            data = {
                "model": "GigaChat",
                "messages": [
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            try:
                resp = requests.post(chat_url, headers=headers, json=data, verify=False, timeout=20)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    answer = resp_json["choices"][0]["message"]["content"].strip()
                    usage = resp_json.get("usage", {})
                else:
                    answer = f"Ошибка API ({resp.status_code}): {resp.text}"
            except Exception as e:
                answer = f"Ошибка запроса: {e}"

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks,
            "top_k": len(retrieved_chunks),
            "usage": usage
        }


def print_result(res: dict):
    """Вывод ответа RAG, информации о токенах и найденных фрагментов."""
    print("=" * 75)
    print(f"❓ ВОПРОС: {res['question']}")
    print("-" * 75)
    print(f"🤖 ОТВЕТ LLM:\n{res['answer']}")

    # Вывод информации о потраченных токенах
    usage = res.get('usage', {})
    if usage and usage.get('total_tokens', 0) > 0:
        total = usage.get('total_tokens', 'N/A')
        prompt_t = usage.get('prompt_tokens', 'N/A')
        comp_t = usage.get('completion_tokens', 'N/A')
        print("-" * 75)
        print(f" [ТОКЕНЫ] Использовано: {total} (Промпт: {prompt_t}, Ответ: {comp_t})")
    elif usage and usage.get('total_tokens', 0) == 0:
        print("-" * 75)
        print(" [ТОКЕНЫ] 0 токенов (запрос отсечен фильтром нерелевантности до вызова LLM)")

    # Проверка: если ответ "не найдено", источники НЕ выводим
    lower_ans = res['answer'].lower()
    is_not_found = (
        "не найдено" in lower_ans or
        "not found in documents" in lower_ans or
        lower_ans.startswith("not found") or
        not res['sources']
    )

    if is_not_found:
        print("-" * 75)
        print("ℹ️ [ИСТОЧНИКИ] Не отображаются, так как релевантной информации по вопросу в базе нет.")
    else:
        print("-" * 75)
        print(f"📚 НАЙДЕННЫЕ ИСТОЧНИКИ (Top-{res['top_k']} чанков, FAISS IndexFlatL2):")
        for i, s in enumerate(res['sources'], 1):
            preview = s['text'][:110] + "..." if len(s['text']) > 110 else s['text']
            print(f"  [{i}] Doc #{s['doc_id']} | Категория: {s['category']} | L2: {s['distance']:.4f}")
            print(f"      Текст: {preview}")
    print("=" * 75)


# =====================================================================
# FastAPI Сервер для RAG-системы
# =====================================================================

_rag_instance: BusinessRAG = None


def get_rag(force_rebuild: bool = False) -> BusinessRAG:
    """Глобальная инициализация модели BERT и индекса FAISS (один раз при старте сервера)."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BusinessRAG(model_name="bert-base-uncased")
        _rag_instance.load_or_build_index(force_rebuild=force_rebuild)
    elif force_rebuild:
        _rag_instance.load_or_build_index(force_rebuild=True)
    return _rag_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели BERT и индекса FAISS при запуске FastAPI сервера."""
    get_rag()
    yield


app = FastAPI(
    title="Business News RAG API",
    description="API-сервер RAG-системы на базе BERT, FAISS и GigaChat для ответов по бизнес-новостям",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Вопрос к нейросети (на английском или русском языке)",
        examples=["What happened with the interest rates decided by the central bank?"],
    )
    top_k: int = Field(
        default=7,
        ge=5,
        le=10,
        description="Количество релевантных фрагментов для поиска в FAISS (от 5 до 10)",
    )


class SourceChunk(BaseModel):
    chunk_id: int
    doc_id: int
    category: str
    text: str
    distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    top_k: int
    usage: dict


@app.get("/", response_class=HTMLResponse, summary="Веб-интерфейс для вопросов и ответов")
def root():
    return """<!DOCTYPE html>
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
                    body: JSON.stringify({ question: q, top_k: 7 })
                });
                const data = await resp.json();
                answerBox.textContent = data.answer || 'Нет ответа';
                if (data.usage && data.usage.total_tokens !== undefined) {
                    tokensBox.textContent = `📊 Токены: всего ${data.usage.total_tokens} (промпт: ${data.usage.prompt_tokens || 0}, ответ: ${data.usage.completion_tokens || 0})`;
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


@app.get("/health", summary="Healthcheck")
def health_check():
    rag = get_rag()
    return {
        "status": "healthy",
        "index_loaded": rag.index is not None,
        "total_chunks": rag.index.ntotal if rag.index is not None else 0,
    }


@app.post("/ask", response_model=AskResponse, summary="Задать вопрос RAG-системе (BERT + FAISS + GigaChat)")
def ask_endpoint(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым.")
    try:
        rag = get_rag()
        result = rag.ask(question=question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {e}")


@app.post("/search", response_model=list[SourceChunk], summary="Поиск релевантных чанков в FAISS без вызова LLM")
def search_endpoint(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Поисковый запрос не может быть пустым.")
    try:
        rag = get_rag()
        return rag.search(query=question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске в FAISS: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG-система на базе BERT и FAISS по бизнес-датасету")
    parser.add_argument('-q', '--query', type=str, help="Текст вопроса для RAG (на английском или русском)")
    parser.add_argument('-k', '--top-k', type=int, default=7, help="Количество чанков для поиска (от 5 до 10, по умолчанию 7)")
    parser.add_argument('-f', '--file', type=str, help="Путь к текстовому файлу с вопросами для пакетного запуска")
    parser.add_argument('-i', '--interactive', action='store_true', help="Запуск в интерактивном консольном режиме")
    parser.add_argument('-s', '--server', action='store_true', help="Запуск FastAPI веб-сервера")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Хост для FastAPI сервера (по умолчанию 0.0.0.0)")
    parser.add_argument('--port', type=int, default=8000, help="Порт для FastAPI сервера (по умолчанию 8000)")
    parser.add_argument('--rebuild-index', action='store_true', help="Принудительно перестроить векторный индекс FAISS")
    args = parser.parse_args()

    if args.server:
        get_rag(force_rebuild=args.rebuild_index)
        print(f"\n🚀 Запуск FastAPI сервера на http://{args.host}:{args.port}")
        print(f"📖 Интерактивная документация (Swagger UI): http://127.0.0.1:{args.port}/docs\n")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    rag = get_rag(force_rebuild=args.rebuild_index)

    if args.query:
        res = rag.ask(args.query, top_k=args.top_k)
        print_result(res)
        return

    if args.file:
        if not os.path.exists(args.file):
            print(f"Файл {args.file} не найден!")
            return
        with open(args.file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"\nЗапуск обработки вопросов из файла: {args.file} ({len(lines)} вопросов)\n")
        for q in lines:
            # Если строка содержит перевод через |, берем только английскую часть для запроса
            query = q.split('|')[0].strip()
            res = rag.ask(query, top_k=args.top_k)
            print_result(res)
            print()
        return

    if args.interactive:
        print("\n" + "=" * 75)
        print("ИНТЕРАКТИВНЫЙ РЕЖИМ RAG-СИСТЕМЫ")
        print("Задавайте вопросы на английском или русском (введите 'exit' для выхода)")
        print("=" * 75)
        while True:
            try:
                user_q = input("\nВведите вопрос: ").strip()
                if user_q.lower() in ['exit', 'quit', 'q']:
                    break
                if not user_q:
                    continue
                res = rag.ask(user_q, top_k=args.top_k)
                print_result(res)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ошибка: {e}")
        return

    # Демонстрационный режим по умолчанию
    print("\nЗапуск демонстрационных вопросов (Top-K = 7):\n")
    demo_qs = [
        "What happened with the interest rates decided by the central bank?",
        "How many electric vehicle makers registered on the online portal for recycling lithium-ion battery waste?",
        "Why did shares of Adani Power climb, and how much stake was acquired by GQG Partners?",
        "Who won the UEFA Champions League football tournament in 2024?"  # Контрольный вопрос
    ]
    for q in demo_qs:
        res = rag.ask(q, top_k=args.top_k)
        print_result(res)
        print()


if __name__ == '__main__':
    main()
