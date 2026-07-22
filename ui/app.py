"""Streamlit UI: uptime, data visualizations, prediction, upload, and retrain.

Talks to the FastAPI service over HTTP. API base URL comes from an env var so
the same UI works locally and when deployed.
"""

import os

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")


def main() -> None:
    """Render the dashboard: status, charts, prediction, upload, retrain button."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
