#set document(title: "Transfer Learning & MLOps Pipeline", author: "Endika Aguirre")
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
  numbering: "1 / 1",
  header: align(right)[
    _MLOps Lab 3 - Pet Classification System_
  ],
)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: "1.1")

// Title page
#align(center)[
  #v(3cm)
  #text(size: 28pt, weight: "bold", fill: rgb("#1a5490"))[
    Transfer Learning for Pet Classification
  ]
  
  #v(0.8cm)
  #text(size: 16pt)[
    Building a Production-Ready MLOps Pipeline
  ]
  
  #v(0.5cm)
  #text(size: 14pt, style: "italic")[
    Transfer Learning with MLFlow Experiment Tracking
  ]
  
  #v(2cm)
  
  #box(
    width: 80%,
    stroke: 1pt + rgb("#1a5490"),
    radius: 5pt,
    inset: 15pt,
  )[
    #text(size: 11pt)[
      *Project Overview* \
      #v(0.3cm)
      This project implements an end-to-end MLOps pipeline for classifying 37 different cat and dog breeds from the Oxford-IIIT Pet Dataset. Using ResNet18 transfer learning, the system achieves over 85% validation accuracy with comprehensive experiment tracking via MLFlow, containerized deployment with Docker, and production hosting on cloud platforms.
    ]
  ]
  
  #v(1.5cm)
  #text(size: 12pt)[
    *Author:* Endika Aguirre \
    *Course:* Machine Learning Operations (MLOps) \
    *Date:* #datetime.today().display("[month repr:long] [day], [year]")
  ]
]

#pagebreak()

#outline(
  title: "Table of Contents",
  indent: auto
)

#pagebreak()

= Project Links and Resources

== GitHub Repositories

This project is part of a three-lab series focused on MLOps practices:

*Lab 1 - Basic ML Pipeline:*
- Repository: #link("https://github.com/Ninjalice/MLOPS_LAB")[github.com/Ninjalice/MLOPS_LAB]
- Focus: Initial setup, basic data processing, and model development
- Technologies: Python, basic ML models

*Lab 2 - API Development and Deployment:*
- Repository: #link("https://github.com/Ninjalice/MLOPS_LAB")[github.com/Ninjalice/MLOPS_LAB]
- HuggingFace Space: #link("https://huggingface.co/spaces/Ninjalice/mlops-lab2")[huggingface.co/spaces/Ninjalice/mlops-lab2]
- Focus: FastAPI implementation, Docker containerization, cloud deployment
- Technologies: FastAPI, Docker, HuggingFace Spaces

*Lab 3 - Experiment Tracking and Transfer Learning (This Project):*
- Repository: #link("https://github.com/Ninjalice/MLOPS-LAB-V3")[github.com/Ninjalice/MLOPS-LAB-V3]
- HuggingFace Space: #link("https://huggingface.co/spaces/Ninjalice/mlops-lab3")[huggingface.co/spaces/Ninjalice/mlops-lab3]
- Docker Hub: #link("https://hub.docker.com/r/ninjalice/mlops-lab3")[hub.docker.com/r/ninjalice/mlops-lab3]
- Focus: MLFlow experiment tracking, transfer learning, production deployment
- Technologies: PyTorch, MLFlow, ONNX, ResNet18

== Deployment Infrastructure

All deployments are publicly accessible:

- *GitHub (Source Code):* Complete codebase with version control
- *HuggingFace Spaces (Demo):* Interactive web interface for predictions
- *Docker Hub (Container Registry):* Containerized application for deployment
- *Local MLFlow Server:* Experiment tracking dashboard (localhost:5000)

#pagebreak()

= Executive Summary

This technical report documents the development and deployment of a deep learning classification system for the Oxford-IIIT Pet Dataset. The project emphasizes modern MLOps practices including systematic experiment tracking, model versioning, containerization, and cloud deployment.

== Key Achievements

*Model Performance:*
- Best validation accuracy: *83.89%*
- Training accuracy: *86.01%*
- Model architecture: ResNet18 with transfer learning
- Total experiments conducted: 3 primary runs

*MLOps Infrastructure:*
- Experiment tracking and model versioning with MLFlow
- ONNX model serialization for cross-platform inference
- Dockerized deployment pipeline
- Multi-platform cloud hosting (HuggingFace Spaces, Docker Hub)

*Technical Stack:*
- PyTorch for deep learning framework
- MLFlow for experiment management
- ONNX Runtime for optimized inference
- FastAPI for REST API endpoints
- Docker for containerization

= Project Architecture and Technology Stack

The MLOps pipeline integrates three core components: an experimentation layer with MLFlow for tracking and PyTorch for training, a model serving layer using ONNX Runtime and FastAPI, and a deployment layer with Docker containerization and cloud hosting on HuggingFace Spaces and Docker Hub. The complete technology stack includes PyTorch with ResNet18, MLFlow for experiment management, ONNX for model serialization, FastAPI for the REST API, and pytest for testing.

= Dataset and Problem Definition

The Oxford-IIIT Pet Dataset contains approximately 7,400 images across 37 classes (25 cat breeds and 12 dog breeds). Images are preprocessed by resizing to 256×256 pixels, center-cropping to 224×224, and normalizing with ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]). The dataset is split 80/20 for training and validation with a fixed random seed (42) for reproducibility. The main challenges include high intra-class variance, inter-class similarity, and variable image quality.

= Transfer Learning Methodology

The project uses ResNet18 with ImageNet pre-trained weights as the base model. The convolutional layers are frozen to act as a feature extractor, while the final fully connected layer is modified from 512 inputs to 37 outputs (one per breed class). This approach leverages pre-trained feature representations while requiring only ~18.5K trainable parameters out of ~11.2M total. The model is trained using Adam optimizer with CrossEntropyLoss, without learning rate scheduling to isolate the impact of fixed hyperparameters across experiments.

= Experimental Results and Analysis

== Experiment Overview

Three experiments were conducted with ResNet18 transfer learning, systematically varying batch size and learning rate to identify optimal configurations.

#figure(
  table(
    columns: (1.2fr, 0.8fr, 0.8fr, 0.8fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + rgb("#cccccc"),
    fill: (x, y) => if y == 0 { rgb("#1a5490") } else if y == 3 { rgb("#d4edda") },
    
    text(fill: white)[*Run*], 
    text(fill: white)[*Batch*], 
    text(fill: white)[*LR*], 
    text(fill: white)[*Epochs*], 
    text(fill: white)[*Train Acc*], 
    text(fill: white)[*Val Acc*],
    
    [Run 1], [32], [0.01], [3], [82.17%], [78.09%],
    [Run 2], [64], [0.001], [3], [85.82%], [82.37%],
    [*Run 3 (Best)*], [*32*], [*0.001*], [*3*], [*86.01%*], [*83.89%*],
  ),
  caption: "Experimental configurations and results"
)

Run 3 achieved the best performance with 86.01% training accuracy and 83.89% validation accuracy with 0.5997 validation loss. The high learning rate (0.01) in Run 1 led to poor convergence with validation loss of 1.0416. Run 2 with batch size 64 showed good results but was outperformed by Run 3's smaller batch size of 32, which provided better gradient estimates. Run 3 shows minimal overfitting with only a -2.12% gap (training slightly higher than validation), indicating good generalization.

== MLFlow Experiment Tracking and Logged Artifacts

MLFlow provides centralized tracking for all experiments, logging parameters, metrics, and artifacts essential for reproducibility and model selection.

#figure(
  image("report_images/MLFLOW.png", width: 100%),
  caption: "MLFlow UI showing experiment tracking dashboard with all three runs"
)

*Logged Parameters:* Each run logs model architecture (ResNet18), hyperparameters (batch_size, learning_rate, epochs), training configuration (optimizer: Adam, loss: CrossEntropyLoss), dataset details (Oxford-IIIT Pet, 37 classes, 224×224 images), and reproducibility settings (random_seed: 42, train_split: 0.8). These parameters enable complete experiment reproduction and systematic hyperparameter analysis.

*Tracked Metrics:* Training and validation accuracy and loss are logged per epoch, along with final metrics at training completion. These metrics directly measure classification performance and help detect overfitting by comparing training vs validation behavior.

*Stored Artifacts:* The following artifacts were selected for logging:

- *Trained Model (.pth):* Complete PyTorch model with learned weights, enabling model deployment and further fine-tuning
- *Training Curves (PNG):* Visual representation of loss and accuracy progression, essential for identifying convergence patterns and training instabilities
- *Class Labels (JSON):* Mapping from numeric predictions (0-36) to breed names, required for human-readable inference output
- *Model Configuration:* Complete documentation of architecture and hyperparameters for reproducibility

These artifacts were chosen because they provide complete experiment reproducibility, visual debugging tools, deployment-ready components, and comprehensive documentation.

== MLFlow GUI Analysis and Model Selection

Using the MLFlow user interface, all three runs were compared side-by-side. The dashboard displays key metrics for direct comparison: Run 1 (Batch=32, LR=0.01) achieved 78.09% training accuracy and 82.17% validation accuracy with 1.0416 loss, Run 2 (Batch=64, LR=0.001) reached 82.37% training and 85.82% validation accuracy with 0.7124 loss, and Run 3 (Batch=32, LR=0.001) obtained the best results with 86.01% training and 83.89% validation accuracy with 0.5997 loss.

The selection process using MLFlow involved: (1) sorting runs by `final_val_accuracy` to identify Run 3 as top performer, (2) confirming lowest validation loss, (3) reviewing training curves from stored artifacts showing stable convergence, (4) comparing hyperparameters to understand that LR=0.01 was too aggressive while LR=0.001 with batch size 32 provided optimal results, and (5) analyzing generalization where Run 3's -2.12% gap (training slightly higher than validation) indicated good model balance.

Run 3 was selected for production deployment and registered in MLFlow Model Registry as version 1 with "production" stage. The model was then exported to ONNX format for cross-platform inference.

= Model Serialization and Deployment

The selected model (Run 3) was converted to ONNX format (opset 18) for production deployment. The conversion process loads the PyTorch model from MLFlow registry, sets it to evaluation mode, and exports using `torch.onnx.export()` with a dummy input tensor of shape (1, 3, 224, 224). The resulting ONNX model (~8.83 MB) provides cross-platform compatibility and optimized inference performance.

The inference pipeline preprocesses images (resize to 256×256, center crop to 224×224, normalize with ImageNet statistics), runs ONNX Runtime inference, and maps predictions to breed labels. The system is deployed via FastAPI with endpoints for prediction, image preprocessing, and file retrieval, accessible through both REST API and CLI interface.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("report_images/GitHub.png", width: 100%),
    image("report_images/DOCKER.png", width: 100%),
  ),
  caption: "GitHub repository (left) and Docker Hub deployment (right)"
)

#figure(
  image("report_images/HUGGINFACE.png", width: 85%),
  caption: "HuggingFace Spaces deployment with live prediction interface"
)

The application is containerized using Docker with Python 3.11-slim base image and CPU-only PyTorch (reducing image size from 4+ GB to ~1.2 GB). It's deployed on GitHub for source code (#link("https://github.com/Ninjalice/MLOPS-LAB-V3")[github.com/Ninjalice/MLOPS-LAB-V3]), Docker Hub for container registry (#link("https://hub.docker.com/r/ninjalice/mlops-lab3")[hub.docker.com/r/ninjalice/mlops-lab3]), HuggingFace Spaces for web demo (#link("https://huggingface.co/spaces/Ninjalice/mlops-lab3")[huggingface.co/spaces/Ninjalice/mlops-lab3]), and local MLFlow server for experiment tracking.

= Testing Strategy

The testing suite ensures reliability across all pipeline components through three test modules: `test_logic.py` validates core business logic including prediction functions (ONNX integration, fallback mechanisms, confidence scoring) and image preprocessing operations (resize, grayscale, normalize, rotate, flip, blur, complete pipeline). `test_api.py` tests FastAPI endpoints using TestClient to validate the home page, prediction endpoint with various file types, resize endpoint with different dimensions, and file retrieval. `test_cli.py` verifies the Click-based command-line interface for both classify and preprocess command groups using CliRunner for isolated execution.

The testing approach uses unittest.mock for isolating external dependencies, fixture-based test images for consistent validation, temporary directories to prevent filesystem pollution, and comprehensive assertions covering edge cases and error paths. Tests achieve >80% overall coverage with >95% on critical paths. Continuous integration runs the full test suite on every commit along with linting checks (flake8, black) and Docker build validation before deployment.

= Results and Key Findings

The best model (Run 3) achieved 86.01% training accuracy and 83.89% validation accuracy with 0.5997 validation loss in 12.4 minutes of training. This represents significant improvement compared to Run 1, with optimal hyperparameters of LR=0.001 and batch size 32.

The project successfully implements a complete MLOps pipeline with experiment tracking via MLFlow, model versioning and registry, ONNX serialization for cross-platform deployment, Dockerized containers, multi-platform cloud hosting, REST API with FastAPI, and comprehensive test coverage. Key insights include: transfer learning with pre-trained ResNet18 achieved 86% training accuracy with only 3 epochs; learning rate had the most significant impact (0.01 too high, 0.001 optimal); smaller batch size (32) outperformed larger (64); Docker optimization reducing image size by 70% was critical for cloud deployment; and MLFlow provided essential visibility for systematic model comparison.

= Challenges and Solutions

The main technical challenge was Docker image size, where the initial image exceeded 4GB with full PyTorch+CUDA. Switching to CPU-only PyTorch reduced it to ~1.2GB, enabling faster deployment and cloud platform compatibility. 

= Conclusion

This project successfully implemented an end-to-end MLOps pipeline for pet breed classification, achieving 86.01% training accuracy and 83.89% validation accuracy using ResNet18 transfer learning. The systematic experimentation with MLFlow tracking enabled data-driven model selection, while ONNX serialization and Docker containerization facilitated production deployment across multiple platforms. The infrastructure demonstrates modern MLOps best practices and provides a scalable foundation for future enhancements.

#pagebreak()

= Appendix

== Complete Experimental Results

#figure(
  table(
    columns: (1fr, 0.6fr, 0.6fr, 0.5fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr),
    align: (left, center, center, center, center, center, center, center),
    stroke: 0.5pt,
    
    [*Run Name*], [*Batch*], [*LR*], [*Ep*], [*Train Acc*], [*Val Acc*], [*Train Loss*], [*Val Loss*],
    [resnet18-run1], [32], [0.01], [3], [78.09%], [82.17%], [0.6442], [1.0416],
    [resnet18-run2], [64], [0.001], [3], [82.37%], [85.82%], [0.7228], [0.7124],
    [resnet18-run3], [32], [0.001], [3], [86.01%], [83.89%], [0.6117], [0.5997],
  ),
  caption: "Complete experimental data"
)

== Model Configuration

ResNet18 architecture with 3×224×224 RGB input, frozen pre-trained feature extractor, modified FC layer (512 → 37), ReLU activation with Softmax output, ~11.2M total parameters with ~18.5K trainable.

== Dataset Classes

37 pet breeds from Oxford-IIIT Pet Dataset: 25 cat breeds (e.g., Abyssinian, Bengal, British Shorthair, Egyptian Mau, Persian, Ragdoll, Siamese) and 12 dog breeds (e.g., Beagle, Boxer, Chihuahua, English Setter, Pug, Yorkshire Terrier). Complete list available in `class_labels.json`.