.PHONY: install test test-int lint format typecheck proto docker-up docker-down clean

install:
	uv sync --extra dev

test:
	uv run pytest tests/unit/ -v

test-int:
	uv run pytest tests/integration/ -v -m integration

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy shared/ data_plane/ control_plane/

proto:
	@echo "TODO: uv run python -m grpc_tools.protoc ..."

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage
