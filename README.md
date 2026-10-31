# Выполнил: Чувакова Анна Сергеевна 
## Группа: МП-302
* ### Датасет:
* Этот датасет представляет собой набор данных для обучения моделей машинного обучения (NLP), который содержит 2000 новостных статей на тему бизнеса и экономики из индийского издания The Indian Express.

---

## Локальный сервер (FastAPI) и Docker-образ

### Структура
* `src/rag.py` — RAG-система (BERT + FAISS + GigaChat), функции `build_index()` и `rag_answer()`.
* `src/app.py` — REST API на FastAPI: `GET /health`, `POST /ask`, веб-страница `/`.
* `test/test_api.py` — тесты API через `requests`.
* `Dockerfile`, `.dockerignore`, `docker-compose.yml` — контейнеризация.

### Локальный запуск
```bash
source venv/bin/activate
pip install -r requirements.txt          # для ноутбуков: pip install -r requirements-dev.txt
uvicorn src.app:app --reload
```
* Swagger UI: http://127.0.0.1:8000/docs, ReDoc: http://127.0.0.1:8000/redoc
* Пример запроса:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What happened with the interest rates decided by the central bank?", "k": 3}'
```
* Тесты: `python test/test_api.py`

### Docker
```bash
docker build -t doc-assistant .          # повторная сборка берёт слои из кэша (CACHED)
docker images doc-assistant              # размер образа

# Секреты НЕ зашиваются в образ — передаются при запуске
docker run --rm -p 8000:8000 \
  -e GIGA_CLIENT_ID=$GIGA_CLIENT_ID -e GIGA_CLIENT_SECRET=$GIGA_CLIENT_SECRET \
  doc-assistant
# или
docker run --rm -p 8000:8000 --env-file .env doc-assistant
```
Через docker compose (с HEALTHCHECK):
```bash
docker compose up -d --build
docker ps                                # статус (healthy)
python test/test_api.py
docker compose down
```
