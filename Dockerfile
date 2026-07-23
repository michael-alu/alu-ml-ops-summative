# API image: used by docker-compose for load testing and by Cloud Run for deploy.
# Cloud Run injects $PORT, so the server binds to ${PORT:-8000}.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-api.txt

COPY src ./src
COPY api ./api
COPY models ./models

RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
