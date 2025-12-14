# MLOps Lab - Image Classification System

![CI/CD Pipeline](https://github.com/Ninjalice/MLOPS-LAB-V3/actions/workflows/ci.yml/badge.svg)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-mlops--lab3-blue?logo=docker)](https://hub.docker.com/r/ninjalice/mlops-lab3)

## 📋 Overview

Image classification system using the **Oxford-IIIT Pet Dataset** (37 cat and dog breeds) with deep learning (ResNet18) and MLOps practices including CI/CD, Docker, and MLflow tracking.

## 🎯 Features

- **Deep Learning Model**: ResNet18 with transfer learning
- **Dataset**: Oxford-IIIT Pet (37 classes) - automatically downloaded
- **MLflow Tracking**: Experiment tracking and model versioning
- **CLI & API**: Multiple interfaces for predictions
- **Docker**: Containerized deployment
- **Gradio GUI**: Web interface on HuggingFace Spaces
- **CI/CD**: Automated testing and deployment via GitHub Actions

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Ninjalice/MLOPS-LAB-V3.git
cd MLOPS-LAB-V3

# Install dependencies
make install
```

### Training

```bash
# Train model (dataset downloads automatically)
make train

# View MLflow UI
make mlflow-ui
```

### Serialization

```bash
# Serialize model from MLflow run
make serialize RUN_ID=your_run_id_here
```

## 🛠️ Make Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make train` | Train model on Oxford-IIIT Pet dataset |
| `make serialize RUN_ID=<id>` | Convert MLflow model to ONNX |
| `make mlflow-ui` | Launch MLflow UI (http://localhost:5000) |
| `make test` | Run tests with coverage |
| `make lint` | Lint code with Pylint |
| `make format` | Format code with Black |
| `make refactor` | Format + Lint |
| `make docker-build` | Build Docker image |
| `make docker-push` | Push image to Docker Hub |
| `make docker-deploy` | Build + Push Docker image |
| `make clean` | Remove generated files and dataset |
| `make all` | Install + Format + Lint + Test |

## 📁 Project Structure

```
MLOPS_V3/
├── api/                    # FastAPI application
├── cli/                    # Command-line interface
├── logic/                  # Classification logic
├── tests/                  # Test suite
├── data/                   # Oxford-IIIT Pet dataset (auto-downloaded)
├── mlruns/                 # MLflow experiment tracking
├── plots/                  # Training plots
├── train_model.py          # Training script
├── serialize_model.py      # Model serialization to ONNX
├── app.py                  # Gradio GUI
├── Dockerfile              # Multi-stage Docker build
├── Makefile               # Build automation
└── pyproject.toml         # Dependencies
```

## 💻 Usage

### Training Model

```bash
# Basic training
make train

# Custom parameters
uv run python train_model.py --epochs 15 --batch-size 64 --learning-rate 0.0001
```

### Using the API

```bash
# Start API server
uv run python -m api.api

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

### Using the CLI

```bash
# Predict image class
uv run python -m cli.cli predict image.jpg

# Preprocess image
uv run python -m cli.cli preprocess input.jpg output.jpg --width 224 --height 224
```

## 🐳 Docker Deployment

```bash
# Build and push to Docker Hub
make docker-deploy

# Run locally
docker run -p 8000:8000 ninjalice/mlops-lab3:latest
```

## 📊 Dataset

**Oxford-IIIT Pet Dataset**: 37 categories of cats and dogs

- **Training**: ~3,680 images (trainval split)
- **Validation**: ~3,669 images (test split)
- **Classes**: Abyssinian, American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle, Bengal, Birman, Bombay, Boxer, British Shorthair, Chihuahua, Egyptian Mau, English Cocker Spaniel, English Setter, German Shorthaired, Great Pyrenees, Havanese, Japanese Chin, Keeshond, Leonberger, Maine Coon, Miniature Pinscher, Newfoundland, Persian, Pomeranian, Pug, Ragdoll, Russian Blue, Saint Bernard, Samoyed, Scottish Terrier, Shiba Inu, Siamese, Sphynx, Staffordshire Bull Terrier, Wheaten Terrier, Yorkshire Terrier

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific tests
uv run pytest tests/test_api.py -v
```

## 🔄 CI/CD Pipeline

GitHub Actions workflow:
1. **CI**: Test, lint, format on pull requests
2. **CD**: Build Docker image and deploy to Render on push to main
3. **HuggingFace**: Deploy Gradio GUI

## 📦 Dependencies

- **PyTorch & Torchvision**: Deep learning framework
- **MLflow**: Experiment tracking
- **FastAPI**: REST API
- **Gradio**: Web GUI
- **ONNX**: Model serialization
- **Click**: CLI framework
- **Pillow**: Image processing

## 👥 Author

- Endika - [Ninjalice](https://github.com/Ninjalice)

## 📄 License

MLOps course project - Universidad Pública de Navarra (UPNA)

---

**Live Deployments:**
- 🌐 API: https://mlops-lab3-cv3j.onrender.com
- 🤗 GUI: https://huggingface.co/spaces/Ninjalice/mlops-lab3
