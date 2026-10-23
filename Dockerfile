FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Копируем исходный код, данные (индекс FAISS) и конфигурацию
COPY src/ ./src/
COPY data/processed/ ./data/processed/

# Предварительная загрузка весов модели bert-base-uncased в образ, чтобы контейнер стартовал мгновенно
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"

EXPOSE 8000

CMD ["uvicorn", "src.rag:app", "--host", "0.0.0.0", "--port", "8000"]
