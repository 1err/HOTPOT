# syntax=docker/dockerfile:1
# Official multi-platform index: Linux amd64 and arm64 use the same source.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOTPOT_RECOGNITION_PROVIDER=openai \
    HOTPOT_OPENAI_MODEL=gpt-5.4 \
    HOTPOT_DATA_DIR=/app/data \
    HOTPOT_MODEL_DIR=/tmp/hotpot-models

WORKDIR /app
COPY requirements-vision.lock ./requirements-vision.lock

# Optional proxy CA is mounted only during installation; ordinary Docker Desktop
# builds need no secret. TLS verification remains enabled in both cases.
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -f /run/secrets/proxy_ca ]; then \
        PIP_CERT=/run/secrets/proxy_ca python -m pip install --no-cache-dir --require-hashes -r requirements-vision.lock; \
    else \
        python -m pip install --no-cache-dir --require-hashes -r requirements-vision.lock; \
    fi

RUN groupadd --gid 1000 hotpot \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin hotpot \
    && install -d -m 0750 -o 1000 -g 1000 /app/data

COPY hotpot/ ./hotpot/
COPY static/ ./static/
COPY run.py ./run.py
# Source files may have private host permissions; runtime can use the host UID.
# Keep code root-owned and readable, and writes confined to the data mount/tmp.
RUN chmod -R a=rX /app/hotpot /app/static \
    && chmod 0644 /app/run.py /app/requirements-vision.lock

USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json, urllib.request; r = json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)); raise SystemExit(0 if r.get('ok') is True else 1)"
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
