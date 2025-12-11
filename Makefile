.PHONY: install lint format test refactor all clean

install:
	@echo "Installing dependencies..."
	uv sync

lint:
	@echo "Linting code with pylint..."
	uv run pylint logic/ cli/ api/ tests/

format:
	@echo "Formatting code with black..."
	uv run black logic/ cli/ api/ tests/

test:
	@echo "Running tests with pytest..."
	uv run pytest tests/ -v --cov=logic --cov=cli --cov=api --cov-report=html --cov-report=term

refactor: format lint
	@echo "Code refactored: formatted and linted!"

all: install format lint test
	@echo "All tasks completed successfully!"

clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"
