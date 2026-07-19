# BUMP — developer entrypoints. `make help` lists targets.
PY ?= python3
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
MODEL := inference/models/beat_cnn.onnx

.DEFAULT_GOAL := help
.PHONY: help setup lint lint-py lint-fe test test-unit test-int model train train-smoke up down logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create a local venv and install shared + all service/dev deps
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install -U pip
	$(VENV_PY) -m pip install -e ./shared
	$(VENV_PY) -m pip install -r ingestion/requirements.txt -r inference/requirements.txt \
		-r api/requirements.txt -r training/requirements.txt
	$(VENV_PY) -m pip install ruff pytest pytest-asyncio
	cd frontend && npm install
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

model: ## Export a shape-correct (untrained) ONNX so inference can boot without training
	$(VENV_PY) training/model.py --out $(MODEL) || $(PY) training/model.py --out $(MODEL)
	@echo "Wrote $(MODEL)"

train: ## Train BeatCNN on MIT-BIH (downloads records), write metrics.json + ONNX
	PYTHONPATH=training $(VENV_PY) training/train.py --out $(MODEL)

train-smoke: ## Fast training smoke run (tiny subset, 1 epoch)
	PYTHONPATH=training $(VENV_PY) training/train.py --smoke --out $(MODEL)

lint: lint-py lint-fe ## Lint Python (ruff) + frontend (eslint)

lint-py: ## Ruff check on all Python
	$(VENV_PY) -m ruff check . || ruff check .

lint-fe: ## ESLint on the frontend
	cd frontend && npm run lint

test: ## Run the full pytest suite
	$(VENV_PY) -m pytest

test-unit: ## Unit tests only (no network/redis)
	$(VENV_PY) -m pytest tests/unit -m "not network"

test-int: ## Integration tests (starts nothing; redis test skips if unreachable)
	$(VENV_PY) -m pytest tests/integration

up: ## Build & start the full stack (Docker Compose)
	docker compose up --build

down: ## Stop the stack and remove volumes
	docker compose down -v

logs: ## Tail service logs
	docker compose logs -f --tail=100

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ **/*.egg-info frontend/dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
