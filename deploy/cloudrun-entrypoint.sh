#!/usr/bin/env bash
# Cloud Run container entrypoint. Mirrors the proven Render start sequence:
# migrate the (persistent, managed) DB, seed/scan demo rows, then serve.
set -euo pipefail

# Apply schema. Idempotent; safe to run on every cold start.
python manage.py migrate --no-input

# Seed demo rows and scan any unresolved ones. The command only seeds when the
# table is empty and only scans unresolved homes, and it swallows per-home scan
# errors — so a missing API key or a flaky scan never blocks the boot.
python manage.py seed_demo --scan || echo "seed_demo --scan skipped/failed (continuing)"

# Cloud Run routes traffic to $PORT (default 8080). One worker keeps memory low
# for the OpenCV/ONNX pipeline; threads handle concurrent requests. A generous
# timeout covers the per-request satellite fetch + inference on a cold home.
exec gunicorn solarleads.wsgi:application \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers "${WEB_CONCURRENCY:-1}" \
    --threads "${GUNICORN_THREADS:-4}" \
    --timeout "${GUNICORN_TIMEOUT:-120}"
