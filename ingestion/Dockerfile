# Build context is the REPO ROOT (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

COPY shared/ ./shared/
RUN pip install --no-cache-dir ./shared

COPY ingestion/requirements.txt ./ingestion/requirements.txt
RUN pip install --no-cache-dir -r ./ingestion/requirements.txt
COPY ingestion/ ./ingestion/
RUN pip install --no-cache-dir ./ingestion

# Cache PhysioNet downloads across restarts if a volume is mounted here.
ENV WFDB_CACHE_DIR=/data/wfdb \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data/wfdb

EXPOSE 9101
CMD ["python", "-m", "bump_ingestion"]
