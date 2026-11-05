ARG BASE_IMAGE=python:3.15.2-slim-bookworm
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_QUANTIZATION=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN pip install --no-deps torch==2.15.0 --index-url ${TORCH_INDEX_URL} \
    && pip install -r requirements-model.txt \
    && if [ "$INSTALL_QUANTIZATION" = "true" ]; then pip install -r requirements-quantization.txt; fi

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /cache/huggingface /models && chown -R appuser:appuser /cache /models
COPY --chown=appuser:appuser issue_filter ./issue_filter
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser wsgi.py gunicorn.conf.py ./
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:app"]
