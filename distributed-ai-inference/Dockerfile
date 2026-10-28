FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# CPU-only PyTorch wheel first (the default Linux wheel pulls GBs of CUDA libraries we don't need).
# The later `pip install -r` sees torch==2.5.1 already satisfied (2.5.1+cpu matches).
COPY requirements.txt .
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app ./app
COPY model ./model

# Train ONCE at build time so every replica started from this image serves the identical artifact.
RUN python -m model.train --output model/artifacts/model.pt

RUN useradd --system --uid 10001 --no-create-home appuser && chown -R appuser /srv
USER appuser

ENV MODEL_PATH=/srv/model/artifacts/model.pt \
    UVICORN_WORKERS=1
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=4s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# One worker per container by default: scale with replicas, and keep Prometheus metrics per-process correct.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS} --no-access-log"]
