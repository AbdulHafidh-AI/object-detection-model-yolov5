import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

st.title("YOLOv5 Object Detection with Bounding Boxes")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize the image to 640x640 before sending it to the API
    image_resized = image.resize((640, 640))
    
    # Save resized image to bytes
    buf = io.BytesIO()
    image_resized.save(buf, format="JPEG")
    byte_image = buf.getvalue()

    # Send image to FastAPI backend
    response = requests.post(
        "http://localhost:8000/predict/",
        files={"file": byte_image}
    )

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image_resized)
        
        for pred in predictions:
            x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
            label = pred["label"]
            confidence = pred["confidence"]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Add label and confidence score
            draw.text((x1, y1), f"{label} {confidence:.2f}", fill="white")
        
        # Display image with bounding boxes
        st.image(image_resized, caption="Processed Image with Bounding Boxes", use_column_width=True)
    else:
        st.write("Error in prediction")
