"""Script to serialize the best MLflow model to ONNX format and save class labels."""

import json
import os
import torch
import mlflow
from mlflow.tracking import MlflowClient


def serialize_model(run_id: str, model_name: str = "model.onnx", labels_name: str = "class_labels.json"):
    """
    Serialize the best MLflow model to ONNX format and save class labels.
    
    Args:
        run_id: The MLflow run ID of the best model version
        model_name: Output filename for the ONNX model (default: "model.onnx")
        labels_name: Output filename for class labels JSON (default: "class_labels.json")
    """
    # Initialize MLflow client
    client = MlflowClient()
    
    print(f"Loading model from run: {run_id}")
    
    # Load the model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Move model to CPU (required for Render deployment)
    print("Moving model to CPU...")
    model = model.to('cpu')
    
    # Set model to evaluation mode
    print("Setting model to evaluation mode...")
    model.eval()
    
    # Create dummy input for ONNX export (assuming 3-channel RGB images of size 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export model to ONNX format
    print(f"Exporting model to ONNX format: {model_name}")
    torch.onnx.export(
        model,
        dummy_input,
        model_name,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {model_name}")
    
    # Download and save class labels
    print(f"Downloading class labels from run: {run_id}")
    try:
        # Download the class labels artifact
        labels_artifact_path = client.download_artifacts(run_id, "class_labels.json")
        
        # Read the downloaded class labels
        with open(labels_artifact_path, 'r', encoding='utf-8') as f:
            class_labels = json.load(f)
        
        # Save class labels to local file
        with open(labels_name, 'w', encoding='utf-8') as f:
            json.dump(class_labels, f, indent=2)
        
        print(f"Class labels saved to {labels_name}")
        print(f"Class labels: {class_labels}")
        
    except Exception as e:
        print(f"Error downloading class labels: {e}")
        print("Creating default class labels...")
        # Fallback to default CIFAR-10 labels if not available
        default_labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        with open(labels_name, 'w', encoding='utf-8') as f:
            json.dump(default_labels, f, indent=2)
        print(f"Default class labels saved to {labels_name}")
    
    print("\nSerialization complete!")
    print(f"✓ Model saved: {model_name}")
    print(f"✓ Labels saved: {labels_name}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python serialize_model.py <run_id>")
        print("Example: python serialize_model.py abc123def456")
        sys.exit(1)
    
    run_id = sys.argv[1]
    serialize_model(run_id)
