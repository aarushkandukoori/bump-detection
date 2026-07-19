# Build context is the REPO ROOT (see docker-compose.yml) so we can vendor the
# shared package into the image.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scipy/onnxruntime ship manylinux wheels.
RUN pip install --no-cache-dir --upgrade pip

# Shared contracts/signal core first (changes least -> better layer caching).
COPY shared/ ./shared/
RUN pip install --no-cache-dir ./shared

# Service.
COPY inference/requirements.txt ./inference/requirements.txt
RUN pip install --no-cache-dir -r ./inference/requirements.txt
COPY inference/ ./inference/
RUN pip install --no-cache-dir ./inference

# Bake in a model if one exists (compose also mounts ./inference/models:/models).
RUN mkdir -p /models && \
    (cp inference/models/beat_cnn.onnx* /models/ 2>/dev/null || \
     echo "no baked model; will use rate-only fallback until one is mounted")

ENV MODEL_PATH=/models/beat_cnn.onnx \
    PYTHONUNBUFFERED=1

EXPOSE 9102
CMD ["python", "-m", "bump_inference"]
