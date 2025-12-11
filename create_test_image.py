"""Script to create a sample test image."""

from PIL import Image
import os

# Create tests directory if it doesn't exist
os.makedirs("tests/test_images", exist_ok=True)

# Create a simple test image
image = Image.new("RGB", (100, 100), color=(255, 0, 0))
image.save("tests/test_images/sample.png")

print("Sample test image created at tests/test_images/sample.png")
