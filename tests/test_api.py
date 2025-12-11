"""Tests for the API module."""

import pytest
from fastapi.testclient import TestClient
from api.api import app
from PIL import Image
import io


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes."""
    image = Image.new("RGB", (100, 100), color="blue")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_home_endpoint(client):
    """Test the home endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_success(client, sample_image_bytes):
    """Test predict endpoint with valid image."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_class" in data
    assert "filename" in data


def test_predict_endpoint_no_file(client):
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_endpoint_invalid_file(client):
    """Test predict endpoint with invalid file."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_resize_endpoint_success(client, sample_image_bytes):
    """Test resize endpoint with valid inputs."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "50", "height": "50"}
    response = client.post("/resize", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["new_size"]["width"] == 50
    assert result["new_size"]["height"] == 50


def test_resize_endpoint_missing_dimensions(client, sample_image_bytes):
    """Test resize endpoint without dimensions."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/resize", files=files)
    assert response.status_code == 422


def test_resize_endpoint_invalid_dimensions(client, sample_image_bytes):
    """Test resize endpoint with invalid dimensions."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "-10", "height": "50"}
    response = client.post("/resize", files=files, data=data)
    assert response.status_code == 400


def test_resize_endpoint_zero_dimensions(client, sample_image_bytes):
    """Test resize endpoint with zero dimensions."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "0", "height": "50"}
    response = client.post("/resize", files=files, data=data)
    assert response.status_code == 400


def test_preprocess_endpoint_with_defaults(client, sample_image_bytes):
    """Test preprocess endpoint with default dimensions."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/preprocess", files=files)

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["new_size"]["width"] == 224
    assert result["new_size"]["height"] == 224
    assert result["new_size"]["mode"] == "RGB"


def test_preprocess_endpoint_with_custom_dimensions(client, sample_image_bytes):
    """Test preprocess endpoint with custom dimensions."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "128", "height": "128"}
    response = client.post("/preprocess", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["new_size"]["width"] == 128
    assert result["new_size"]["height"] == 128


def test_preprocess_endpoint_invalid_dimensions(client, sample_image_bytes):
    """Test preprocess endpoint with invalid dimensions."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "-128", "height": "128"}
    response = client.post("/preprocess", files=files, data=data)
    assert response.status_code == 400


def test_classify_and_resize_endpoint_success(client, sample_image_bytes):
    """Test classify_and_resize endpoint with valid inputs."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "64", "height": "64"}
    response = client.post("/classify_and_resize", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert "predicted_class" in result
    assert result["new_size"]["width"] == 64
    assert result["new_size"]["height"] == 64


def test_classify_and_resize_endpoint_missing_dimensions(client, sample_image_bytes):
    """Test classify_and_resize endpoint without dimensions."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/classify_and_resize", files=files)
    assert response.status_code == 422


def test_classify_and_resize_endpoint_invalid_file(client):
    """Test classify_and_resize endpoint with invalid file."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    data = {"width": "64", "height": "64"}
    response = client.post("/classify_and_resize", files=files, data=data)
    assert response.status_code == 400


def test_grayscale_image_conversion(client):
    """Test API handles grayscale images correctly."""
    # Create a grayscale image
    image = Image.new("L", (100, 100), color=128)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/preprocess", files=files)

    assert response.status_code == 200
    result = response.json()
    assert result["new_size"]["mode"] == "RGB"
