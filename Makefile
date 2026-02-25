.PHONY: install run test lint docker-build docker-up docker-down retrain

# ── Local development ───────────────────────────────────────

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

lint:
	ruff check . --fix
	mypy . --ignore-missing-imports

# ── Docker ──────────────────────────────────────────────────

docker-build:
	docker build -t bug-priority-engine:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

# ── Model management ────────────────────────────────────────

retrain:
	@echo "Triggering manual retrain via API..."
	curl -s -X POST http://localhost:8000/api/v1/retrain \
		-H "X-Retrain-API-Key: $${RETRAIN_API_KEY}" | python -m json.tool

health:
	curl -s http://localhost:8000/api/v1/health | python -m json.tool
