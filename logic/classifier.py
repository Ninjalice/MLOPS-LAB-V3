"""Image classification and preprocessing logic."""

import random
from typing import Tuple
from PIL import Image


# Define available class names for classification
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


def predict_class(image: Image.Image) -> str:
    """
    Predict the class of a given image.

    In this initial version, the class is randomly chosen from available classes.

    Args:
        image: PIL Image object to classify

    Returns:
        str: Predicted class name
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    # For now, randomly select a class
    predicted_class = random.choice(CLASS_NAMES)
    return predicted_class


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
