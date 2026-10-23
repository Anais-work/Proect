"""
Модуль RAG-системы (Retrieval-Augmented Generation) на основе бизнес-датасета.
Реализован в строгом соответствии с методическим документом RAG.pdf:
1. Модель эмбеддингов: BERT (bert-base-uncased) для англоязычного текста.
2. Чанкинг документов: длина 500 символов, перекрытие (overlap) 50 символов.
3. Векторный индекс: FAISS (faiss.IndexFlatL2) по евклидову расстоянию.
4. ANN-поиск: извлечение 5-10 наиболее релевантных чанков (k регулируется).
5. Сборка контекста + строгий промпт против галлюцинаций + вызов LLM (GigaChat API) + возврат источников.
"""

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


def main():
    parser = argparse.ArgumentParser(description="RAG-система на базе BERT и FAISS по бизнес-датасету")
    parser.add_argument('-q', '--query', type=str, help="Текст вопроса для RAG (на английском или русском)")
    parser.add_argument('-k', '--top-k', type=int, default=7, help="Количество чанков для поиска (от 5 до 10, по умолчанию 7)")
    parser.add_argument('-f', '--file', type=str, help="Путь к текстовому файлу с вопросами для пакетного запуска")
    parser.add_argument('-i', '--interactive', action='store_true', help="Запуск в интерактивном консольном режиме")
    parser.add_argument('--rebuild-index', action='store_true', help="Принудительно перестроить векторный индекс FAISS")
    args = parser.parse_args()

    rag = BusinessRAG(model_name="bert-base-uncased")
    rag.load_or_build_index(force_rebuild=args.rebuild_index)

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
