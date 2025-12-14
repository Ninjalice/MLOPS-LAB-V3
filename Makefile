.PHONY: install lint format test refactor all clean train serialize mlflow-ui docker-build docker-push docker-deploy

install:
	@echo "Installing dependencies..."
	uv sync

lint:
	@echo "Linting code with pylint..."
	uv run pylint --rcfile=.pylintrc --ignore-patterns=test_.*\.py logic/ cli/ api/

format:
	@echo "Formatting code with black..."
	uv run black logic/ cli/ api/ tests/

test:
	@echo "Running tests with pytest..."
	uv run pytest tests/ -v --cov=logic --cov=cli --cov=api --cov-report=html --cov-report=term

train:
	@echo "Training model on Oxford-IIIT Pet dataset with MLflow tracking..."
	@echo "Dataset will be downloaded automatically on first run..."
	uv run python train_model.py --epochs 3 --batch-size 64 --learning-rate 0.001 --data-dir ./data

serialize:
	@echo "Please provide RUN_ID as argument: make serialize RUN_ID=your_run_id"
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Error: RUN_ID not provided"; \
		echo "Usage: make serialize RUN_ID=abc123def456"; \
		exit 1; \
	fi
	@echo "Serializing model from run $(RUN_ID)..."
	uv run python serialize_model.py $(RUN_ID)

mlflow-ui:
	@echo "Starting MLflow UI on http://localhost:5000..."
	uv run mlflow ui

docker-build:
	@echo "Building Docker image: ninjalice/mlops-lab3:latest"
	docker build -t ninjalice/mlops-lab3:latest .
	@echo "✓ Docker image built successfully!"

docker-push:
	@echo "Pushing Docker image to Docker Hub..."
	docker push ninjalice/mlops-lab3:latest
	@echo "✓ Image pushed to Docker Hub!"

docker-deploy: docker-build docker-push
	@echo "✓ Docker image built and pushed successfully!"
	@echo "Image: ninjalice/mlops-lab3:latest"

refactor: format lint
	@echo "Code refactored: formatted and linted!"

all: install format lint test
	@echo "All tasks completed successfully!"

clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	rm -rf data/oxford-iiit-pet
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"
