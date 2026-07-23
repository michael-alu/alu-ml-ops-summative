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

- GitHub repo: TBD
- Live URL: TBD
- Video demo (YouTube): TBD

## Project structure

```
alu-ml-ops-summative/
├── README.md
├── requirements.txt
├── Dockerfile                 # API image (load test + deploy)
├── docker-compose.yml         # nginx + scalable api replicas
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
│   └── generate_insights.py   # builds dataset insight charts for the UI
├── nginx/
│   └── nginx.conf             # load balancer for the container scaling test
├── locust/
│   └── locustfile.py          # flood test
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

To be filled in from the notebook run.

| Metric | Value |
|---|---|
| Accuracy | TBD |
| Loss | TBD |
| Precision (macro) | TBD |
| Recall (macro) | TBD |
| F1 (macro) | TBD |

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

### Flood test results

To be filled in after running the test across 1, 2, and 3 containers.

| Containers | Users | RPS | Median latency (ms) | p95 latency (ms) | Failures |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |
