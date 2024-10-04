from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and resize image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((640, 640))  # Resize to 640x640

    # Inference
    results = model(image)

    # Extract predictions (bounding box, label, and confidence)
    boxes = results.xyxy[0].tolist()  # x1, y1, x2, y2, confidence, class
    labels = results.names  # Class labels
    
    # Prepare response
    predictions = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        predictions.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf,
            "label": labels[int(cls)]
        })

    return {"predictions": predictions}
