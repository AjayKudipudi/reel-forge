.PHONY: install dev test lint type fmt smoke clean

install:
	pip install -e .

dev:
	pip install -e .[dev]
	pre-commit install

test:
	pytest tests/unit tests/contract tests/integration

lint:
	ruff check insta_influencer tests

type:
	mypy insta_influencer

fmt:
	ruff check --fix insta_influencer tests
	ruff format insta_influencer tests

smoke:
	pytest -m smoke tests/smoke

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
