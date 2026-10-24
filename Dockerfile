# syntax=docker/dockerfile:1
# API image (also used by the `ui` service in docker-compose.yml).
#
#   docker compose build
#
# No torch: the query encoder runs on onnxruntime (ENCODER=onnx), from the model repo's own ONNX
# export at a pinned revision, checked by sha256 (the same pins as src/search.py ONNX_FILES;
# tests/test_search.py checks they match). Layers go from least to most often changed: the model
# files, then requirements-api.txt, then the code, so editing src/ only rebuilds the last layer.
# Data is not baked in: compose mounts ./data/processed read-only at /app/data/processed (the
# relative paths in src/interfaces.py).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENCODER=onnx

# Not root. data/cache/llm is the parser's response cache.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/processed /app/data/cache /app/models \
    && chown -R app:app /app
WORKDIR /app

ADD --chown=app:app \
    --checksum=sha256:6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452 \
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx \
    models/all-MiniLM-L6-v2-onnx/model.onnx
ADD --chown=app:app \
    --checksum=sha256:be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037 \
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json \
    models/all-MiniLM-L6-v2-onnx/tokenizer.json

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY --chown=app:app db/ db/
COPY --chown=app:app src/ src/

USER app
EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
