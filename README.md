---
title: Sound Classification MLOps
emoji: 🔊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Sound Classification MLOps Pipeline

An end-to-end MLOps pipeline that classifies environmental sounds from `.wav`
audio. It covers the full machine learning cycle: data acquisition, preprocessing,
an optimized CNN, evaluation, a prediction API, a dashboard UI, bulk data upload,
one-click retraining, containerized deployment, and a Locust load test across
different numbers of Docker containers.

The dataset is the **ESC-10** subset of ESC-50 (10 balanced environmental sound
classes). Audio is converted to log-mel-spectrograms and classified by a compact,
regularized CNN.

## Links

- GitHub repo: TBD (add your repo URL)
- Live URL (Hugging Face Space): TBD (add after deploying)
- Video demo (YouTube): TBD (add your demo link)

## Project structure

```
alu-ml-ops-summative/
├── README.md
├── Dockerfile                 # HF Space image (API + UI on port 7860)
├── docker-compose.yml         # nginx + scalable api replicas (load test)
├── start.sh                   # launches API + UI in the HF image
├── requirements.txt           # full stack (notebook + dev)
├── requirements-app.txt       # deployed app runtime deps
├── requirements-api.txt       # slim api-image deps (load test)
├── docker/
│   └── Dockerfile.api         # API-only image used by docker-compose
├── notebook/
│   └── sound_classification.ipynb
├── src/
│   ├── preprocessing.py       # wav -> mel-spectrogram (single source of truth)
│   ├── model.py               # CNN build / train / save / load
│   ├── prediction.py          # single-file prediction
│   └── retrain.py             # retraining pipeline
├── api/
│   └── main.py                # FastAPI: predict, upload, retrain, health
├── ui/
│   └── app.py                 # Streamlit dashboard
├── scripts/
│   ├── generate_insights.py   # builds dataset insight charts for the UI
│   └── run_load_test.sh       # automated Locust run across container counts
├── nginx/
│   └── nginx.conf             # load balancer for the container scaling test
├── locust/
│   └── locustfile.py          # flood test
├── load_test_results/         # Locust CSV results
├── assets/                    # insight charts + a sample clip
├── data/
│   ├── raw/                   # downloaded ESC-50 (gitignored)
│   ├── train/                 # wav files by class (gitignored)
│   ├── test/                  # wav files by class (gitignored)
│   └── uploads/               # bulk uploads for retraining (gitignored)
└── models/                    # exported model + class names
```

## Model summary

- Input: 5 second clip resampled to 22050 Hz, converted to a 128-band
  log-mel-spectrogram normalized to [0, 1].
- Architecture: 3 convolutional blocks (Conv2D + BatchNorm + MaxPool), global
  average pooling, dense head with dropout.
- Optimization techniques: L2 regularization, dropout, BatchNorm, Adam, early
  stopping on validation accuracy, and ReduceLROnPlateau.
- Data: augmented with added noise, time shift, and pitch shift.

### Evaluation metrics (held-out fold 5 test set)

| Metric | Value |
|---|---|
| Accuracy | 0.863 |
| Loss | 0.443 |
| Precision (macro) | 0.872 |
| Recall (macro) | 0.863 |
| F1 (macro) | 0.863 |

## Setup

Requires Python 3.10.

```bash
pip install -r requirements.txt
```

### 1. Train the model (notebook)

Open and run every cell in `notebook/sound_classification.ipynb`. It downloads
ESC-50, preprocesses audio, trains the CNN, evaluates it, and writes
`models/sound_model.h5` and `models/class_names.json`.

### 2. Generate dataset insight charts (for the UI)

```bash
python scripts/generate_insights.py
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs are at `http://localhost:8000/docs`.

### 4. Run the UI

```bash
export API_BASE_URL=http://localhost:8000
streamlit run ui/app.py
```

The dashboard shows model uptime, dataset visualizations, single-clip prediction,
bulk upload, and a retraining button.

## Deploy to Hugging Face Spaces

The root `Dockerfile` runs the API and the UI together on port 7860 for a public
Hugging Face Space. The Space config lives in the YAML frontmatter at the top of
this README (`sdk: docker`, `app_port: 7860`).

1. Create a new Space at https://huggingface.co/new-space and pick the **Docker** SDK.
2. Push this repository to the Space:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```

3. Hugging Face builds the image and serves the dashboard at the Space URL. The
   trained model ships in the repo (`models/`), so no extra download is needed.

Locally the same image runs with:

```bash
docker build -t sound-hf .
docker run -p 7860:7860 sound-hf
# open http://localhost:7860
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Uptime, model status, class list |
| POST | `/predict` | Predict the class of one uploaded wav |
| POST | `/upload` | Save bulk wav files under a class label for retraining |
| POST | `/retrain` | Retrain on uploaded data and reload the model |

## Retraining flow

1. Upload one or more `.wav` files under a chosen class label (UI or `/upload`).
   Files are saved to `data/uploads/<label>/`.
2. Trigger retraining (UI button or `/retrain`). The saved model is loaded as a
   pretrained base, fine-tuned on the preprocessed uploads, evaluated, and saved.
3. The API reloads the new model automatically.

## Load test with Locust and Docker

The container scaling test runs locally with nginx load-balancing across N api
replicas.

```bash
# Build and start with 1 replica
docker compose up --build --scale api=1

# In another terminal, run Locust against nginx
locust -f locust/locustfile.py --host http://localhost:8080
```

Repeat with `--scale api=2` and `--scale api=3`, running the same Locust test each
time, and record latency and requests per second.

The whole run is automated:

```bash
bash scripts/run_load_test.sh
```

### Flood test results

Conditions: 50 concurrent users, 60 seconds per run, spawn rate 10/s, on a 4-CPU
Colima VM. Requests hit nginx, which round-robins across the api replicas.

| Containers | Users | RPS | Median latency (ms) | p95 latency (ms) | Requests | Failures |
|---|---|---|---|---|---|---|
| 1 | 50 | 11.6 | 2800 | 17000 | 685 | 0 |
| 2 | 50 | 11.3 | 2500 | 21000 | 666 | 0 |
| 3 | 50 | 14.6 | 1600 | 19000 | 861 | 0 |

Scaling from 1 to 3 containers cut median latency by about 43% (2800 to 1600 ms)
and raised throughput by about 26% (11.6 to 14.6 RPS), with zero failures at every
level. The gains are modest because the Colima VM has only 4 CPUs shared across all
replicas plus nginx and Locust, and TensorFlow inference is CPU-bound, so replicas
beyond the available cores give diminishing returns. The high absolute latencies
are the expected effect of 50 users flooding CPU-bound inference at once.
