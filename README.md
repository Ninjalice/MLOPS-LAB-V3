# MLOps Labs 1 & 2 - Image Classification System

![CI/CD Pipeline](https://github.com/Ninjalice/MLOPS_LAB/actions/workflows/ci.yml/badge.svg)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-mlops--lab2-blue?logo=docker)](https://hub.docker.com/r/yourusername/mlops-lab2)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/yourusername/mlops-lab2)

## 📋 Project Overview

This project implements an **Image Classification System** with a complete **CI/CD pipeline** using GitHub Actions for Continuous Integration and Continuous Delivery. The system provides functionality for image classification, resizing, and preprocessing through multiple interfaces:

- **Logic Module**: Core functionality for image processing and classification
- **CLI**: Command-line interface for local operations
- **API**: RESTful API built with FastAPI for web-based interactions
- **Docker**: Containerized deployment for reproducibility
- **Gradio GUI**: User-friendly web interface hosted on HuggingFace Spaces
- **Cloud Deployment**: API hosted on Render

## 🎯 Features

### Image Classification
- Random class prediction from predefined categories
- Support for 10 classes: cat, dog, bird, fish, horse, deer, frog, car, airplane, ship

### Image Processing
- **Resize**: Change image dimensions to any size
- **RGB Conversion**: Convert images to RGB mode
- **Preprocessing**: Combined RGB conversion and resizing (default: 224x224)
- **Image Info**: Get image dimensions and color mode

## 🏗️ Project Structure

```
MLOPS_LAB/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline configuration
├── api/
│   ├── __init__.py
│   └── api.py                  # FastAPI application
├── cli/
│   ├── __init__.py
│   └── cli.py                  # Command-line interface
├── logic/
│   ├── __init__.py
│   └── classifier.py           # Core logic for classification
├── templates/
│   └── home.html               # API homepage template
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # API tests
│   ├── test_cli.py            # CLI tests
│   └── test_logic.py          # Logic tests
├── .gitignore
├── Makefile                    # Build automation
├── pyproject.toml              # Project dependencies and configuration
├── README.md                   # This file
└── Lab1.pdf                    # Assignment document
```

## � Docker Deployment

### Building the Docker Image

```bash
docker build -t mlops-lab2 .
```

### Running the Container

```bash
docker run -p 8000:8000 mlops-lab2
```

The API will be available at `http://localhost:8000`.

### Multi-Stage Build Details

The Dockerfile uses a multi-stage build for optimization:
- **Base Stage**: Python 3.13-slim foundation with system dependencies
- **Builder Stage**: Installs Python dependencies using uv
- **Runtime Stage**: Minimal final image (< 200MB) with only runtime dependencies

## ☁️ Cloud Deployments

### Render (API Hosting)

The API is automatically deployed to Render via GitHub Actions on every push to `main`.

- **Live API**: https://mlops-lab2.onrender.com
- **Auto-Deploy**: Triggered by GitHub Actions workflow
- **Setup Required**: 
  1. Create a Render web service
  2. Connect to your Docker Hub repository
  3. Add `RENDER_DEPLOY_HOOK_KEY` to GitHub secrets

### HuggingFace Spaces (GUI)

Interactive Gradio GUI hosted on HuggingFace Spaces.

- **Live Demo**: https://huggingface.co/spaces/yourusername/mlops-lab2
- **Features**: Upload images, get predictions, view results
- **Auto-Deploy**: Synced via GitHub Actions on push to `main`
- **Setup Required**:
  1. Create a HuggingFace Space
  2. Add `HF_USERNAME` and `HF_TOKEN` to GitHub secrets

## �🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ninjalice/MLOPS_LAB.git
cd MLOPS_LAB
```

2. Install dependencies:
```bash
make install
# or manually:
uv sync
```

## 💻 Usage

### Command Line Interface (CLI)

The CLI provides several commands for image processing:

#### Predict Image Class
```bash
uv run python -m cli.cli predict <image_path>
```

#### Resize Image
```bash
uv run python -m cli.cli resize <image_path> <width> <height> <output_path>
```

#### Preprocess Image
```bash
uv run python -m cli.cli preprocess <image_path> <output_path> --width 224 --height 224
```

#### Convert to RGB
```bash
uv run python -m cli.cli to-rgb <image_path> <output_path>
```

#### Get Image Information
```bash
uv run python -m cli.cli info <image_path>
```

### API

Start the FastAPI server:
```bash
uv run python -m api.api
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `GET /` - Homepage with API documentation
- `GET /health` - Health check endpoint
- `POST /predict` - Classify an image
- `POST /resize` - Resize an image
- `POST /preprocess` - Preprocess an image (RGB + resize)
- `POST /classify_and_resize` - Combined classification and resizing

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## 🧪 Testing

Run all tests:
```bash
make test
# or manually:
uv run pytest tests/ -v --cov=logic --cov=cli --cov=api
```

## 🛠️ Development

### Available Make Commands

```bash
make install    # Install dependencies
make format     # Format code with Black
make lint       # Lint code with Pylint
make test       # Run tests with Pytest
make refactor   # Format and lint code
make all        # Run all tasks (install, format, lint, test)
make clean      # Clean up generated files
```

### Code Quality

- **Formatting**: Black (line length: 100)
- **Linting**: Pylint
- **Testing**: Pytest with coverage reporting

## 🔄 CI/CD Pipeline

This project uses GitHub Actions for complete CI/CD automation with three main jobs:

### 1. Build & Test (CI)
1. Checks out the code
2. Sets up Python environment
3. Installs dependencies with uv
4. Formats code with Black
5. Lints code with Pylint
6. Runs tests with Pytest
7. Uploads coverage reports

### 2. Deploy API (CD)
1. Builds Docker image with multi-stage optimization
2. Pushes image to Docker Hub
3. Triggers Render deployment via webhook

### 3. Deploy HuggingFace Space (CD)
1. Checks out orphan `hf-space` branch
2. Copies Gradio app and requirements
3. Pushes to HuggingFace Spaces repository

### Pipeline Triggers
- **Push to `main`**: Full CI/CD (test + deploy)
- **Pull requests to `main` or `develop`**: CI only (test)

### Required GitHub Secrets
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub personal access token
- `RENDER_DEPLOY_HOOK_KEY`: Render deploy hook URL key
- `HF_USERNAME`: HuggingFace username
- `HF_TOKEN`: HuggingFace write access token

## 📦 Dependencies

### Core Dependencies
- **click**: Command-line interface framework
- **fastapi**: Modern web framework for APIs
- **uvicorn**: ASGI server
- **jinja2**: Template engine for HTML
- **httpx**: HTTP client
- **pillow**: Image processing library
- **python-multipart**: File upload support

### Development Dependencies
- **pylint**: Code linting
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting

## 📝 Assignment Requirements

### Lab 1: Continuous Integration ✅
✅ Machine learning project structure with incremental methodology  
✅ Logic module with image classification and preprocessing  
✅ Command-line interface (CLI) using Click  
✅ RESTful API using FastAPI  
✅ Comprehensive test suite for all components (41 tests, 94% coverage)  
✅ Virtual environment management with uv  
✅ HTML homepage for API  
✅ Makefile for automation (install, lint, format, test)  
✅ CI pipeline with GitHub Actions  
✅ Status badge in README  
✅ Proper project documentation  

### Lab 2: Continuous Delivery ✅
✅ Multi-stage Dockerfile (base, builder, runtime)  
✅ Docker image optimization (< 200MB)  
✅ Healthcheck in Docker container  
✅ API deployment to Render web service  
✅ Gradio GUI on HuggingFace Spaces  
✅ Automated deployments via GitHub Actions  
✅ Docker Hub integration  
✅ GitHub secrets for secure credentials  
✅ CD pipeline triggered on push to main  
✅ Documentation for all deployment platforms  

## 👥 Authors

- Endika - [Ninjalice](https://github.com/Ninjalice)

## 📄 License

This project is part of the MLOps course at UPNA (Universidad Pública de Navarra).

## 🙏 Acknowledgments

- Based on the MLOps Lab 1 demo repository: [JoseanSanz/MLOps-Lab1-demo](https://github.com/JoseanSanz/MLOps-Lab1-demo)
- UPNA MLOps Course instructors

---

