#!/usr/bin/env bash
# Launches the API in the background and the Streamlit UI in the foreground.
# The UI reads API_BASE_URL (set to the internal API) to reach predictions.
set -e

uvicorn api.main:app --host 127.0.0.1 --port 8000 &

streamlit run ui/app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
