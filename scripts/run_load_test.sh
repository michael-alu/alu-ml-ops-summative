#!/usr/bin/env bash
# Runs the Locust flood test against the API across different container counts.
# Builds the image once, then for each replica count: scales up, waits until the
# model is loaded, runs a fixed Locust load, and tears down. Results go to
# load_test_results/ as CSV plus a printed summary.
set -euo pipefail

cd "$(dirname "$0")/.."

COUNTS=(1 2 3)
USERS=50
SPAWN_RATE=10
DURATION=60s
HOST=http://localhost:8080
OUT_DIR=load_test_results

mkdir -p "$OUT_DIR"

echo "Building api image ..."
docker compose build

for n in "${COUNTS[@]}"; do
    echo ""
    echo "=== Scaling api to $n container(s) ==="
    docker compose up -d --scale "api=$n"

    echo "Waiting for the model to load ..."
    for _ in $(seq 1 60); do
        if curl -sf "$HOST/health" | grep -q '"model_loaded":true'; then
            echo "Ready."
            break
        fi
        sleep 2
    done

    echo "Running Locust ($USERS users, $DURATION) ..."
    python3 -m locust -f locust/locustfile.py --host "$HOST" \
        --headless -u "$USERS" -r "$SPAWN_RATE" -t "$DURATION" \
        --csv "$OUT_DIR/containers_$n" --only-summary

    docker compose down
done

echo ""
echo "=== Summary (aggregated rows) ==="
for n in "${COUNTS[@]}"; do
    row=$(tail -n +2 "$OUT_DIR/containers_${n}_stats.csv" | grep -i "Aggregated" || true)
    echo "containers=$n: $row"
done
echo ""
echo "Full stats in $OUT_DIR/*_stats.csv"
