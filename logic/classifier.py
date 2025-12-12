"""Image classification and preprocessing logic."""

import json
import os
from typing import Tuple, List
import numpy as np
from PIL import Image
import onnxruntime as ort


# Define available class names for classification (legacy)
CLASS_NAMES = [
    "cat",
    "dog",
    "bird",
    "fish",
    "horse",
    "deer",
    "frog",
    "car",
    "airplane",
    "ship",
]


class ONNXClassifier:
    """ONNX-based image classifier."""
    
    def __init__(self, model_path: str = "model.onnx", labels_path: str = "class_labels.json"):
        """
        Initialize the ONNX classifier.
        
        Args:
            model_path: Path to the ONNX model file
            labels_path: Path to the class labels JSON file
        """
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        # Get input name
        self.input_name = self.session.get_inputs()[0].name
        
        # Load class labels
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.class_labels = json.load(f)
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to 224x224 (standard for many models)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Normalize: (img / 255 - mean) / std
        img_array = img_array / 255.0
        img_array = (img_array - mean) / std
        
        # Change from HWC to CHW format (Height, Width, Channels -> Channels, Height, Width)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Image.Image) -> str:
        """
        Predict the class of an image.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Predicted class label as string
        """
        # Preprocess the image
        input_data = self.preprocess(image)
        
        # Create inputs dictionary
        inputs = {self.input_name: input_data}
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Get logits (first output)
        logits = outputs[0]
        
        # Get predicted class index
        predicted_idx = np.argmax(logits, axis=1)[0]
        
        # Return class label
        return self.class_labels[predicted_idx]


# Create global classifier instance (will be initialized when files exist)
_classifier = None


def get_classifier() -> ONNXClassifier:
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        model_path = "model.onnx"
        labels_path = "class_labels.json"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please run serialize_model.py to create the ONNX model."
            )
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}. "
                "Please run serialize_model.py to create the class labels file."
            )
        
        _classifier = ONNXClassifier(model_path, labels_path)
    
    return _classifier


def predict_class(image: Image.Image) -> str:
    """
    Predict the class of a given image using ONNX model.

    Args:
        image: PIL Image object to classify

    Returns:
        str: Predicted class name
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    try:
        classifier = get_classifier()
        return classifier.predict(image)
    except FileNotFoundError:
        # Fallback to random prediction if model files don't exist
        import random
        import warnings
        warnings.warn("Model files not found. Using random prediction as fallback.")
        return random.choice(CLASS_NAMES)


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Resize an image to the specified dimensions.

    Args:
        image: PIL Image object to resize
        width: Target width in pixels
        height: Target height in pixels

    Returns:
        Image.Image: Resized PIL Image object
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")

    resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
    return resized_image


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert an image to RGB mode.

    Args:
        image: PIL Image object to convert

    Returns:
        Image.Image: RGB PIL Image object
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def normalize_image(image: Image.Image) -> Tuple[int, int, str]:
    """
    Get normalized information about an image.

    Args:
        image: PIL Image object to analyze

    Returns:
        Tuple containing (width, height, mode)
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    return image.size[0], image.size[1], image.mode


def preprocess_image(
    image: Image.Image, target_width: int = 224, target_height: int = 224
) -> Image.Image:
    """
    Preprocess an image: convert to RGB and resize.

    Args:
        image: PIL Image object to preprocess
        target_width: Target width (default: 224)
        target_height: Target height (default: 224)

    Returns:
        Image.Image: Preprocessed PIL Image object
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    # Convert to RGB
    rgb_image = convert_to_rgb(image)

    # Resize to target dimensions
    preprocessed_image = resize_image(rgb_image, target_width, target_height)

    return preprocessed_image
