#!/usr/bin/env python3
"""Command Line Interface for image classification."""

import click
from PIL import Image
from pathlib import Path
from logic.classifier import (
    predict_class,
    resize_image,
    preprocess_image,
    convert_to_rgb,
    normalize_image,
)


@click.group()
def cli():
    """Image Classification CLI - A tool for image preprocessing and classification."""
    pass


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def predict(image_path):
    """
    Predict the class of an image.

    IMAGE_PATH: Path to the image file
    """
    try:
        image = Image.open(image_path)
        predicted_class = predict_class(image)
        click.echo(f"Predicted class: {predicted_class}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.argument("output_path", type=click.Path())
def resize(image_path, width, height, output_path):
    """
    Resize an image to specified dimensions.

    IMAGE_PATH: Path to the input image file
    WIDTH: Target width in pixels
    HEIGHT: Target height in pixels
    OUTPUT_PATH: Path to save the resized image
    """
    try:
        image = Image.open(image_path)
        resized = resize_image(image, width, height)
        resized.save(output_path)
        click.echo(f"Image resized to {width}x{height} and saved to {output_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--width", default=224, help="Target width (default: 224)")
@click.option("--height", default=224, help="Target height (default: 224)")
def preprocess(image_path, output_path, width, height):
    """
    Preprocess an image (convert to RGB and resize).

    IMAGE_PATH: Path to the input image file
    OUTPUT_PATH: Path to save the preprocessed image
    """
    try:
        image = Image.open(image_path)
        preprocessed = preprocess_image(image, width, height)
        preprocessed.save(output_path)
        click.echo(f"Image preprocessed (RGB, {width}x{height}) and saved to {output_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def to_rgb(image_path, output_path):
    """
    Convert an image to RGB mode.

    IMAGE_PATH: Path to the input image file
    OUTPUT_PATH: Path to save the RGB image
    """
    try:
        image = Image.open(image_path)
        rgb_image = convert_to_rgb(image)
        rgb_image.save(output_path)
        click.echo(f"Image converted to RGB and saved to {output_path}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def info(image_path):
    """
    Get information about an image.

    IMAGE_PATH: Path to the image file
    """
    try:
        image = Image.open(image_path)
        width, height, mode = normalize_image(image)
        click.echo(f"Image information:")
        click.echo(f"  Size: {width}x{height}")
        click.echo(f"  Mode: {mode}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
