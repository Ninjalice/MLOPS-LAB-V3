"""Gradio application for Image Classification."""

import gradio as gr
import requests
import io

# API URL - Replace with your Render deployment URL
API_URL = "https://mlops-lab2-fq77.onrender.com"  # Update this after deploying to Render


def predict_image(image):
    """
    Predict the class of an uploaded image using the API.
    
    Args:
        image: PIL Image object from Gradio
        
    Returns:
        str: Predicted class label
    """
    if image is None:
        return "Please upload an image first."
    
    try:
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        # Send request to API
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            predicted_class = result.get("predicted_class", "Unknown")
            return f"🎯 Predicted Class: **{predicted_class}**"
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Error: Request timeout. The API might be starting up (cold start). Please try again."
    except requests.exceptions.ConnectionError:
        return "🔌 Error: Could not connect to the API. Please check if the API is running."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Image Classification - MLOps Lab 2") as demo:
    gr.Markdown(
        """
        # 🖼️ Image Classification System
        ### MLOps Lab 2 - Continuous Delivery Pipeline
        
        Upload an image to get a predicted class label using our deep learning model.
        
        **Supported Classes:** cat, dog, bird, fish, horse, deer, frog, car, airplane, ship
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Upload Image",
                height=400
            )
            predict_btn = gr.Button("🔮 Predict Class", variant="primary", size="lg")
            
        with gr.Column():
            output_text = gr.Markdown(
                label="Prediction Result",
                value="Upload an image and click **Predict Class** to see the result."
            )
    
    gr.Markdown(
        """
        ---
        ### ℹ️ About
        
        This application is part of the MLOps course at UPNA. It demonstrates:
        - 🐳 **Docker** containerization
        - 🚀 **Render** cloud deployment
        - 🤗 **HuggingFace Spaces** for GUI hosting
        - ⚙️ **GitHub Actions** for CI/CD
        
        **Note:** The API runs on a free tier and may experience cold starts (15-30 seconds on first request).
        
        ---
        **Repository:** [GitHub - MLOPS_LAB](https://github.com/Ninjalice/MLOPS_LAB)  
        **API Status:** [Check Health](""" + API_URL + """/health)
        """
    )
    
    # Connect the button to the prediction function
    predict_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=output_text
    )
    
    # Also trigger on image upload
    input_image.change(
        fn=predict_image,
        inputs=input_image,
        outputs=output_text
    )


demo.launch()