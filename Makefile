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
	uv run python -m grpc_tools.protoc -I shared/proto \
		--python_out=shared/proto --pyi_out=shared/proto \
		--grpc_python_out=shared/proto shared/proto/kv_cache.proto
	@# Fix generated import to use package-qualified path
	sed -i '' 's/^import kv_cache_pb2/from shared.proto import kv_cache_pb2/' shared/proto/kv_cache_pb2_grpc.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage
