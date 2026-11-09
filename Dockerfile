# Solar Lead Finder — container image for Google Cloud Run.
# Python 3.12 matches the runtime the app is validated against (see render.yaml).
FROM python:3.12-slim

# opencv-python-headless still needs glib at runtime. Everything else installs
# from manylinux wheels (numpy, onnxruntime, pillow, psycopg[binary]), so no
# compiler/toolchain is needed — keeps the image small.
# The package was renamed libglib2.0-0 -> libglib2.0-0t64 in Debian 13 (trixie),
# so try both to stay compatible across slim base revisions.
RUN apt-get update \
    && (apt-get install -y --no-install-recommends libglib2.0-0 \
        || apt-get install -y --no-install-recommends libglib2.0-0t64) \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install deps first so the layer caches across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Collect static at build time; WhiteNoise serves them in production. A throwaway
# SECRET_KEY lets the command run under DEBUG=False without baking a real secret
# into the image (the real one is injected at runtime by Cloud Run).
RUN SECRET_KEY=build-only-not-used DJANGO_DEBUG=False \
    python manage.py collectstatic --no-input

RUN chmod +x /app/deploy/cloudrun-entrypoint.sh
CMD ["/app/deploy/cloudrun-entrypoint.sh"]
