# Sound Classification MLOps Pipeline

ALU Machine Learning Operations Summative. An end-to-end MLOps pipeline that
classifies environmental sounds from `.wav` audio, with prediction, bulk upload,
retraining, a dashboard UI, and load testing.

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
│   ├── model.py               # CNN build/train/save/load
│   ├── prediction.py          # single-file prediction
│   └── retrain.py             # retraining pipeline
├── api/
│   └── main.py                # FastAPI: predict, upload, retrain, health
├── ui/
│   └── app.py                 # Streamlit dashboard
├── nginx/
│   └── nginx.conf             # load balancer for container scaling test
├── locust/
│   └── locustfile.py          # flood test
├── data/
│   ├── train/
│   ├── test/
│   └── uploads/               # bulk uploads for retraining
└── models/                    # exported model files
```

## Setup

Steps will be finalized once the app is built. Planned flow:

```bash
pip install -r requirements.txt
```

## Deliverables checklist

- [ ] Evaluation notebook (>=4 metrics, optimized model)
- [ ] Prediction (single wav -> class)
- [ ] Bulk upload + retraining trigger
- [ ] Dashboard UI (uptime, visualizations, train/retrain)
- [ ] Docker + Locust flood test across container counts
- [ ] Public deployment URL
- [ ] Video demo (YouTube)

## Links

- GitHub repo: TBD
- Live URL: TBD
- Video demo: TBD

## Flood test results

To be added after the Locust run across 1, 2, and 3 containers.
