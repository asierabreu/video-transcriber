COMPOSE      := docker compose
COMPOSE_GPU  := $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml
TIER         ?= all
FILE         ?=
API_URL      ?= http://localhost:8000

.PHONY: help install dev test test-e2e lint typecheck licenses api-key \
        build-cpu build-gpu up-cpu up-gpu down logs models models-local \
        export-models import-models fixtures bench

help:
	@echo "Local:   install dev test test-e2e lint typecheck licenses models-local bench api-key"
	@echo "Docker:  build-cpu build-gpu up-cpu up-gpu down logs models export-models import-models"
	@echo "Vars:    TIER=$(TIER) (all|cuda-large|cuda-medium|cuda-small|apple|cpu)  FILE=video for bench"

# ---------- local development (macOS / Linux, native) ----------
install:
	uv sync --extra ml --extra dev

dev:
	uv run honcho start -f Procfile.dev

test:
	uv run --extra dev pytest

test-e2e:
	uv run --env-file .env --extra ml --extra dev pytest -m e2e -rs

lint:
	uv run --extra dev ruff check src tests scripts

typecheck:
	uv run --extra dev mypy

licenses:
	uv run --extra ml --extra dev pip-licenses --partial-match \
		--fail-on="GPL;Affero;Non-Commercial;NonCommercial" --ignore-packages video-transcriber

api-key:
	@python3 -c "import secrets; print(secrets.token_urlsafe(32))"

models-local:
	uv run --env-file .env --extra ml python scripts/download_models.py --tier $(TIER)

fixtures:
	./scripts/gen_fixtures.sh

bench:
	@test -n "$(FILE)" || (echo "usage: make bench FILE=path/to/video.mp4" && exit 1)
	uv run --env-file .env python scripts/bench.py "$(FILE)" --url $(API_URL)

# ---------- Docker ----------
build-cpu:
	$(COMPOSE) --profile cpu build

build-gpu:
	$(COMPOSE_GPU) --profile gpu build

up-cpu:
	$(COMPOSE) --profile cpu up -d --build

up-gpu:
	$(COMPOSE_GPU) --profile gpu up -d --build

down:
	$(COMPOSE) --profile cpu --profile gpu --profile tools down

logs:
	$(COMPOSE) --profile cpu --profile gpu logs -f --tail=100

models:
	$(COMPOSE) --profile tools up -d ollama
	$(COMPOSE) --profile tools run --rm models --tier $(TIER)

export-models:
	./scripts/export_models.sh models-bundle.tar.gz

import-models:
	./scripts/import_models.sh models-bundle.tar.gz
