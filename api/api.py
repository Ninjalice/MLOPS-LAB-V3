#!/usr/bin/env python3
"""FastAPI application for image classification."""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from PIL import Image
import io
from logic.classifier import (
    predict_class,
    resize_image,
    preprocess_image,
    normalize_image,
)


app = FastAPI(
    title="Image Classification API",
    description="API for image classification and preprocessing",
    version="1.0.0",
)

# Configure templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home endpoint - serves the HTML homepage.
    """
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.

    Args:
        file: Image file to classify

    Returns:
        JSON with predicted class
    """
    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Predict the class
        predicted_class = predict_class(image)

        return JSONResponse(
            content={
                "success": True,
                "predicted_class": predicted_class,
                "filename": file.filename,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/resize")
async def resize(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
):
    """
    Resize an uploaded image.

    Args:
        file: Image file to resize
        width: Target width in pixels
        height: Target height in pixels

    Returns:
        JSON with resized image information
    """
    try:
        # Validate dimensions
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400, detail="Width and height must be positive integers"
            )

        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Get original size
        original_width, original_height, mode = normalize_image(image)

        # Resize the image
        resized = resize_image(image, width, height)

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "original_size": {"width": original_width, "height": original_height},
                "new_size": {"width": width, "height": height},
                "mode": mode,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/preprocess")
async def preprocess(
    file: UploadFile = File(...),
    width: int = Form(224),
    height: int = Form(224),
):
    """
    Preprocess an uploaded image (convert to RGB and resize).

    Args:
        file: Image file to preprocess
        width: Target width in pixels (default: 224)
        height: Target height in pixels (default: 224)

    Returns:
        JSON with preprocessed image information
    """
    try:
        # Validate dimensions
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400, detail="Width and height must be positive integers"
            )

        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Get original info
        original_width, original_height, original_mode = normalize_image(image)

        # Preprocess the image
        preprocessed = preprocess_image(image, width, height)

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "original_size": {
                    "width": original_width,
                    "height": original_height,
                    "mode": original_mode,
                },
                "new_size": {"width": width, "height": height, "mode": "RGB"},
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/classify_and_resize")
async def classify_and_resize(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
):
    """
    Classify and resize an image in one request.

    Args:
        file: Image file to process
        width: Target width in pixels
        height: Target height in pixels

    Returns:
        JSON with predicted class and resized image information
    """
    try:
        # Validate dimensions
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400, detail="Width and height must be positive integers"
            )

        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Predict class
        predicted_class = predict_class(image)

        # Get original info
        original_width, original_height, mode = normalize_image(image)

        # Resize the image
        resized = resize_image(image, width, height)

        return JSONResponse(
            content={
                "success": True,
                "predicted_class": predicted_class,
                "filename": file.filename,
                "original_size": {"width": original_width, "height": original_height},
                "new_size": {"width": width, "height": height},
                "mode": mode,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
