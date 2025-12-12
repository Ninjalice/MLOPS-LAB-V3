"""Tests for the logic module."""

import os
import pytest
from PIL import Image
from logic.classifier import (
    predict_class,
    resize_image,
    convert_to_rgb,
    normalize_image,
    preprocess_image,
    CLASS_NAMES,
)


def test_predict_class_with_valid_image():
    """Test prediction with a valid image."""
    image = Image.new("RGB", (100, 100), color="red")
    predicted = predict_class(image)
    assert predicted in CLASS_NAMES


def test_predict_class_with_invalid_input():
    """Test prediction with invalid input."""
    with pytest.raises(ValueError, match="Input must be a PIL Image object"):
        predict_class("not_an_image")


def test_resize_image_with_valid_dimensions():
    """Test resizing with valid dimensions."""
    image = Image.new("RGB", (100, 100), color="blue")
    resized = resize_image(image, 50, 50)
    assert resized.size == (50, 50)


def test_resize_image_with_invalid_image():
    """Test resizing with invalid image."""
    with pytest.raises(ValueError, match="Input must be a PIL Image object"):
        resize_image("not_an_image", 50, 50)


def test_resize_image_with_zero_dimensions():
    """Test resizing with zero dimensions."""
    image = Image.new("RGB", (100, 100), color="blue")
    with pytest.raises(ValueError, match="Width and height must be positive integers"):
        resize_image(image, 0, 50)


def test_resize_image_with_negative_dimensions():
    """Test resizing with negative dimensions."""
    image = Image.new("RGB", (100, 100), color="blue")
    with pytest.raises(ValueError, match="Width and height must be positive integers"):
        resize_image(image, -10, 50)


def test_convert_to_rgb_with_rgb_image():
    """Test RGB conversion with already RGB image."""
    image = Image.new("RGB", (100, 100), color="green")
    rgb_image = convert_to_rgb(image)
    assert rgb_image.mode == "RGB"


def test_convert_to_rgb_with_grayscale_image():
    """Test RGB conversion with grayscale image."""
    image = Image.new("L", (100, 100), color=128)
    rgb_image = convert_to_rgb(image)
    assert rgb_image.mode == "RGB"


def test_convert_to_rgb_with_invalid_input():
    """Test RGB conversion with invalid input."""
    with pytest.raises(ValueError, match="Input must be a PIL Image object"):
        convert_to_rgb("not_an_image")


def test_normalize_image():
    """Test getting normalized image information."""
    image = Image.new("RGB", (200, 150), color="yellow")
    width, height, mode = normalize_image(image)
    assert width == 200
    assert height == 150
    assert mode == "RGB"


def test_normalize_image_with_invalid_input():
    """Test normalize with invalid input."""
    with pytest.raises(ValueError, match="Input must be a PIL Image object"):
        normalize_image("not_an_image")


def test_preprocess_image_with_defaults():
    """Test preprocessing with default dimensions."""
    image = Image.new("RGB", (100, 100), color="purple")
    preprocessed = preprocess_image(image)
    assert preprocessed.size == (224, 224)
    assert preprocessed.mode == "RGB"


def test_preprocess_image_with_custom_dimensions():
    """Test preprocessing with custom dimensions."""
    image = Image.new("L", (100, 100), color=128)
    preprocessed = preprocess_image(image, 128, 128)
    assert preprocessed.size == (128, 128)
    assert preprocessed.mode == "RGB"


def test_preprocess_image_with_invalid_input():
    """Test preprocessing with invalid input."""
    with pytest.raises(ValueError, match="Input must be a PIL Image object"):
        preprocess_image("not_an_image")


def test_class_names_not_empty():
    """Test that CLASS_NAMES is not empty."""
    assert len(CLASS_NAMES) > 0


def test_class_names_are_strings():
    """Test that all class names are strings."""
    assert all(isinstance(name, str) for name in CLASS_NAMES)


def test_onnx_model_file_exists():
    """Test that the ONNX model file exists."""
    model_path = "model.onnx"
    assert os.path.exists(model_path), (
        f"ONNX model file not found: {model_path}. "
        "Please run serialize_model.py to create the model before containerization."
    )


def test_class_labels_file_exists():
    """Test that the class labels JSON file exists."""
    labels_path = "class_labels.json"
    assert os.path.exists(labels_path), (
        f"Class labels file not found: {labels_path}. "
        "Please run serialize_model.py to create the labels file before containerization."
    )


def test_class_labels_file_valid():
    """Test that the class labels file contains valid JSON."""
    labels_path = "class_labels.json"
    if os.path.exists(labels_path):
        import json
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        assert isinstance(labels, list), "Class labels should be a list"
        assert len(labels) > 0, "Class labels should not be empty"
        assert all(isinstance(label, str) for label in labels), "All labels should be strings"
