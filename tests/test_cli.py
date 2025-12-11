"""Tests for the CLI module."""

import pytest
from click.testing import CliRunner
from cli.cli import cli
from PIL import Image
from pathlib import Path
import tempfile
import os


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_image():
    """Create a temporary sample image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image = Image.new("RGB", (100, 100), color="red")
        image.save(f.name)
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


def test_predict_command(runner, sample_image):
    """Test the predict command."""
    result = runner.invoke(cli, ["predict", sample_image])
    assert result.exit_code == 0
    assert "Predicted class:" in result.output


def test_predict_command_with_nonexistent_file(runner):
    """Test predict command with nonexistent file."""
    result = runner.invoke(cli, ["predict", "nonexistent.png"])
    assert result.exit_code != 0


def test_resize_command(runner, sample_image):
    """Test the resize command."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
        output_path = output_file.name

    try:
        result = runner.invoke(cli, ["resize", sample_image, "50", "50", output_path])
        assert result.exit_code == 0
        assert "Image resized" in result.output
        assert os.path.exists(output_path)

        # Verify the resized image
        resized = Image.open(output_path)
        assert resized.size == (50, 50)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_resize_command_with_invalid_dimensions(runner, sample_image):
    """Test resize command with invalid dimensions."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
        output_path = output_file.name

    try:
        result = runner.invoke(cli, ["resize", sample_image, "-10", "50", output_path])
        assert result.exit_code != 0
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_preprocess_command_with_defaults(runner, sample_image):
    """Test the preprocess command with default dimensions."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
        output_path = output_file.name

    try:
        result = runner.invoke(cli, ["preprocess", sample_image, output_path])
        assert result.exit_code == 0
        assert "Image preprocessed" in result.output
        assert os.path.exists(output_path)

        # Verify the preprocessed image
        preprocessed = Image.open(output_path)
        assert preprocessed.size == (224, 224)
        assert preprocessed.mode == "RGB"
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_preprocess_command_with_custom_dimensions(runner, sample_image):
    """Test the preprocess command with custom dimensions."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
        output_path = output_file.name

    try:
        result = runner.invoke(
            cli, ["preprocess", sample_image, output_path, "--width", "128", "--height", "128"]
        )
        assert result.exit_code == 0
        assert "128x128" in result.output

        # Verify the preprocessed image
        preprocessed = Image.open(output_path)
        assert preprocessed.size == (128, 128)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_to_rgb_command(runner, sample_image):
    """Test the to-rgb command."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
        output_path = output_file.name

    try:
        result = runner.invoke(cli, ["to-rgb", sample_image, output_path])
        assert result.exit_code == 0
        assert "converted to RGB" in result.output

        # Verify the RGB image
        rgb_image = Image.open(output_path)
        assert rgb_image.mode == "RGB"
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_info_command(runner, sample_image):
    """Test the info command."""
    result = runner.invoke(cli, ["info", sample_image])
    assert result.exit_code == 0
    assert "Image information:" in result.output
    assert "Size:" in result.output
    assert "Mode:" in result.output


def test_info_command_with_nonexistent_file(runner):
    """Test info command with nonexistent file."""
    result = runner.invoke(cli, ["info", "nonexistent.png"])
    assert result.exit_code != 0
