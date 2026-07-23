"""Locust flood test against the /predict endpoint.

Run against nginx (http://localhost:8080) while scaling api replicas to record
latency and response time at different Docker container counts.
"""

from pathlib import Path

from locust import HttpUser, between, task

SAMPLE_WAV: Path = Path(__file__).resolve().parent.parent / "assets" / "sample.wav"
SAMPLE_BYTES: bytes = SAMPLE_WAV.read_bytes()


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        """Send a sample wav to the prediction endpoint."""
        files = {"file": ("sample.wav", SAMPLE_BYTES, "audio/wav")}
        self.client.post("/predict", files=files)
