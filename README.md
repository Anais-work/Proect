# Выполнил: Чувакова Анна Сергеевна
## Группа: МП-302

###Датасет:
Этот набор данных представляет собой набор данных для моделей машинного обучения (NLP), который содержит 2000 новостных статей на темы бизнеса и экономики из индийского издания The Indian Express.

Бизнес-ассистент на базе **PyTorch**, **BERT + FAISS (RAG)**, **MLflow** и **ИИ-агентов (Native / JSON / LangGraph)** с REST API на **FastAPI**.

---

## Структура проекта

```text
Proect/
├── src/
│   ├── app.py              # FastAPI сервер (/health, /ask, /classify, /agent, UI)
│   ├── rag.py              # RAG-система (BERT + FAISS + GigaChat)
│   ├── classifier.py       # PyTorch MLP-классификатор + инференс
│   ├── train_mlflow.py     # Обучение с MLflow Tracking & Model Registry (Staging)
│   ├── prepare.py          # Подготовка и очистка данных
│   └── agents/             # ИИ-агенты (3 подхода) и инструменты
│       ├── tools.py        # search, calc, classify_topic + TOOL_SCHEMAS + run_tool
│       ├── llm_client.py   # Клиент GigaChat API
│       ├── native_agent.py # Вариант 1: Native tool_calls
│       ├── json_agent.py   # Вариант 2: SYSTEM-промпт + extract_json
│       └── framework_agent.py # Вариант 3: LangGraph StateGraph
├── scripts/
│   └── compare_agents.py   # Сравнение 3 подходов на составном запросе
├── test/
│   ├── test_api.py         # Интеграционные тесты живого HTTP-сервера
│   ├── test_agents.py      # Юнит- и mock-тесты инструментов и 3 агентов
│   └── test_classifier.py  # Тесты классификатора и FastAPI (TestClient)
├── model/                  # Сохранённая лучшая модель классификатора
├── data/processed/         # Подготовленный датасет и FAISS-индекс
├── notebook/               # Исследовательские ноутбуки
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── requirements-dev.txt
```

---

## 1. Локальный запуск сервера (FastAPI)

```bash
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.app:app --reload
```

* Веб-интерфейс: http://127.0.0.1:8000/
* Swagger UI: http://127.0.0.1:8000/docs
* Основные эндпоинты:
  * `GET /health` — проверка состояния
  * `POST /ask` — RAG-ответ по базе бизнес-новостей (`{"question": "...", "k": 3}`)
  * `POST /classify` — классификация темы текста PyTorch-моделью (`{"text": "..."}`)
  * `POST /agent` — запуск ИИ-агента (`{"question": "...", "mode": "native|json|framework"}`)

---

## 2. MLOps и MLflow (Задание №1)

Запуск подбора гиперпараметров, логирования параметров/метрик/графиков/конфига/тегов и регистрации лучшей модели в стадии `Staging`:

```bash
python -m src.train_mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

После запуска откройте http://127.0.0.1:5000 (эксперимент `doc-classifier`, модель `doc-classifier` в Model Registry со стадией `Staging`).

---

## 3. ИИ-агенты (Задание №2)

Сравнение трёх подходов (`native`, `json`, `framework`) на составном запросе с вызовом нескольких инструментов:

```bash
python -m scripts.compare_agents --mock   # детерминированный прогон без внешнего API
python -m scripts.compare_agents          # с реальным GigaChat API (требует .env)
```

---

## 4. Автотесты

```bash
pytest -v
```

При поднятом сервере (`uvicorn src.app:app`) дополнительно можно прогнать HTTP-тесты:

```bash
python test/test_api.py
```

---

## 5. Docker

```bash
docker compose up -d --build
docker ps
docker compose down
```
