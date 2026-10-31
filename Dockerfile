# 1. Базовый образ с Docker Hub
FROM python:3.11-slim

# 2. Кэширование слоёв: сначала только зависимости — чтобы слой
#    кэшировался и не пересобирался зря при изменении кода
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Предзагрузка весов bert-base-uncased в образ (отдельный кэшируемый слой),
# чтобы контейнер не скачивал модель при каждом старте
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"

# Готовый индекс FAISS (только нужные файлы, остальное отсекает .dockerignore)
COPY data/processed/ ./data/processed/

# Теперь копируем остальной код
COPY src/ ./src/

EXPOSE 8000

# Проверка здоровья сервиса (в slim-образе нет curl, используем python)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Секреты (GIGA_CLIENT_ID / GIGA_CLIENT_SECRET) НЕ зашиваем в образ —
# передаются при запуске: docker run -e ... или --env-file .env
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
