"""Locust flood test against the /predict endpoint.

Run against nginx (http://localhost:8080) while scaling api replicas to record
latency and response time at different Docker container counts.
"""

from locust import HttpUser, between, task


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        """Send a sample wav to the prediction endpoint."""
        raise NotImplementedError
