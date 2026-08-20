FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=10000 \
    RIPPLE_DATA_DIR=/data/ripple \
    RIPPLE_CACHE_DIR=/data/cache \
    RIPPLE_MAX_UPLOAD_MIB=256

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      libegl1 \
      libgl1 \
      libgles2 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY cutdetect ./cutdetect
COPY scripts/container_feature_smoke.py ./scripts/container_feature_smoke.py

RUN python -m pip install --no-cache-dir ".[features]" \
    && mkdir -p /app/.cutdetect/models \
    && curl -fsSL \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
      -o /app/.cutdetect/models/face_landmarker.task \
    && python scripts/container_feature_smoke.py /app/.cutdetect/models/face_landmarker.task \
    && addgroup --system ripple \
    && adduser --system --ingroup ripple ripple \
    && mkdir -p /data/ripple /data/cache \
    && chown -R ripple:ripple /data

USER ripple
EXPOSE 10000

CMD ["python", "-m", "cutdetect.pipeline.hosting"]
