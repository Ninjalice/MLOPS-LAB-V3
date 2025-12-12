"""Convert Oxford-IIIT Pet Dataset annotations to JSON format."""

import os
import json
import argparse
from pathlib import Path


def parse_image_list(list_file, class_labels):
    """
    Parse the image list file and create annotations.
    
    Format of list file:
    image_name class_id species breed_id
    Example: Abyssinian_1 1 1 1
    """
    annotations = []
    
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            image_name = parts[0]
            class_id = int(parts[1]) - 1  # Convert to 0-indexed
            
            # Extract class name from image name (everything before the last underscore and number)
            # Example: "Abyssinian_100" -> "Abyssinian"
            class_name_parts = image_name.split('_')[:-1]
            class_name = '_'.join(class_name_parts)
            
            # Make sure class name matches our labels
            if class_name in class_labels:
                annotations.append({
                    'filename': f"{image_name}.jpg",
                    'class': class_name,
                    'class_id': class_labels.index(class_name)
                })
    
    return annotations


def create_train_val_split(trainval_file, test_file, class_labels):
    """
    Create train and validation splits from trainval and test files.
    We'll use trainval for training and test for validation.
    """
    print("Creating training annotations...")
    train_annotations = parse_image_list(trainval_file, class_labels)
    
    print("Creating validation annotations...")
    val_annotations = parse_image_list(test_file, class_labels)
    
    return train_annotations, val_annotations


def main(args):
    """Main function to convert annotations."""
    
    # Load class labels
    print(f"Loading class labels from {args.class_labels}")
    with open(args.class_labels, 'r', encoding='utf-8') as f:
        class_labels = json.load(f)
    
    print(f"Found {len(class_labels)} classes")
    
    # Create annotations directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create train/val split
    train_annotations, val_annotations = create_train_val_split(
        args.trainval_file,
        args.test_file,
        class_labels
    )
    
    # Save annotations
    train_output = os.path.join(args.output_dir, 'train.json')
    val_output = os.path.join(args.output_dir, 'val.json')
    
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2)
    
    print(f"\n✓ Created {len(train_annotations)} training annotations -> {train_output}")
    print(f"✓ Created {len(val_annotations)} validation annotations -> {val_output}")
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_annotations)}")
    print(f"  Validation samples: {len(val_annotations)}")
    print(f"  Total samples: {len(train_annotations) + len(val_annotations)}")
    print(f"  Number of classes: {len(class_labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pet Dataset annotations to JSON")
    
    parser.add_argument("--trainval-file", type=str, 
                       default="train_data/annotations/trainval.txt",
                       help="Path to trainval.txt file")
    parser.add_argument("--test-file", type=str,
                       default="train_data/annotations/test.txt",
                       help="Path to test.txt file")
    parser.add_argument("--class-labels", type=str,
                       default="class_labels.json",
                       help="Path to class_labels.json file")
    parser.add_argument("--output-dir", type=str,
                       default="train_data/annotations",
                       help="Output directory for JSON annotations")
    
    args = parser.parse_args()
    
    main(args)
