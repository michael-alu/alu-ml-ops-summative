# Hugging Face Space image: runs the FastAPI backend and the Streamlit UI together.
# Streamlit is exposed on port 7860 (the HF default); uvicorn stays internal on 8000.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as user 1000; set it up before copying files.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    API_BASE_URL=http://127.0.0.1:8000
WORKDIR $HOME/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-app.txt

COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user ui ./ui
COPY --chown=user models ./models
COPY --chown=user assets ./assets
COPY --chown=user start.sh ./start.sh

RUN mkdir -p data/uploads

EXPOSE 7860

CMD ["bash", "start.sh"]
