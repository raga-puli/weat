# API image (also used by the `ui` service in docker-compose.yml).
#
#   docker compose build
#
# Layers go from least to most often changed: CPU-only torch, then requirements.txt, then the code,
# so editing src/ only rebuilds the last layer. Data is not baked in: compose mounts
# ./data/processed read-only at /app/data/processed (the relative paths in src/interfaces.py).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

# Not root. /cache/huggingface is a named volume in compose; creating it here with the right owner
# makes a fresh volume inherit that ownership. data/cache/llm is the parser's response cache.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/processed /app/data/cache /cache/huggingface \
    && chown -R app:app /app /cache
WORKDIR /app

# CPU-only torch first, from PyTorch's index; otherwise sentence-transformers pulls the CUDA build
# (several GB). Pinned to the version the team's .venv uses.
ARG TORCH_VERSION=2.14.0
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app db/ db/
COPY --chown=app:app src/ src/

USER app
EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
