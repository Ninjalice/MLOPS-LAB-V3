"""Train an image classification model using transfer learning and MLflow tracking."""

import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    """Custom Dataset for loading images with annotations."""
    
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file: Path to annotations JSON file
            img_dir: Directory with all the images
            transform: Optional transform to be applied on a sample
        """
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        
        # Create mapping from class name to index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.get_classes())}
        
    def get_classes(self):
        """Get unique class labels from annotations."""
        classes = set()
        for item in self.annotations:
            classes.add(item['class'])
        return sorted(list(classes))
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, annotation['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.class_to_idx[annotation['class']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_transforms():
    """Create data augmentation and normalization transforms."""
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def create_model(num_classes, pretrained=True, freeze_features=True):
    """
    Create a ResNet18 model with transfer learning.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        freeze_features: Freeze feature extraction layers
        
    Returns:
        model: PyTorch model
    """
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=pretrained)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='plots'):
    """Plot and save training history."""
    os.makedirs(save_path, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plot_file = os.path.join(save_path, 'training_history.png')
    plt.savefig(plot_file)
    plt.close()
    
    return plot_file


def main(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transforms, val_transforms = create_data_transforms()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CustomImageDataset(
        annotations_file=args.train_annotations,
        img_dir=args.train_images,
        transform=train_transforms
    )
    
    val_dataset = CustomImageDataset(
        annotations_file=args.val_annotations,
        img_dir=args.val_images,
        transform=val_transforms
    )
    
    # Get class labels
    class_labels = train_dataset.get_classes()
    num_classes = len(class_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_labels}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes, pretrained=True, freeze_features=args.freeze_features)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters() if args.freeze_features else model.parameters(),
                          lr=args.learning_rate)
    
    # Start MLflow run
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("freeze_features", args.freeze_features)
        mlflow.log_param("model_architecture", "resnet18")
        
        # Save class labels as artifact
        os.makedirs("results", exist_ok=True)
        class_labels_path = "results/class_labels.json"
        with open(class_labels_path, 'w', encoding='utf-8') as f:
            json.dump(class_labels, f, indent=2)
        mlflow.log_artifact(class_labels_path)
        
        # Training loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                mlflow.log_metric("best_val_acc", best_val_acc)
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        # Plot training history
        plot_file = plot_training_history(train_losses, val_losses, train_accs, val_accs)
        mlflow.log_artifact(plot_file)
        
        # Log the model
        print("\nLogging model to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\n✓ Training complete!")
        print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    
    # Data arguments
    parser.add_argument("--train-annotations", type=str, default="train_data/annotations/train.json",
                       help="Path to training annotations JSON")
    parser.add_argument("--train-images", type=str, default="train_data/images",
                       help="Path to training images directory")
    parser.add_argument("--val-annotations", type=str, default="train_data/annotations/val.json",
                       help="Path to validation annotations JSON")
    parser.add_argument("--val-images", type=str, default="train_data/images",
                       help="Path to validation images directory")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--freeze-features", action="store_true", default=True,
                       help="Freeze feature extraction layers")
    
    # MLflow arguments
    parser.add_argument("--experiment-name", type=str, default="image-classification",
                       help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default="resnet18-transfer-learning",
                       help="MLflow run name")
    
    args = parser.parse_args()
    
    main(args)
