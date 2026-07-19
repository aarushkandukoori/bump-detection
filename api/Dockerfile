# Build context is the REPO ROOT (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

COPY shared/ ./shared/
RUN pip install --no-cache-dir ./shared

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r ./api/requirements.txt
COPY api/ ./api/
RUN pip install --no-cache-dir ./api

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "bump_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
