from flask import Flask, request, jsonify
import os
import requests
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# 🔽 Step 1: Download the model if it's not available
def download_model():
    url = "https://drive.google.com/uc?export=download&id=12CxkuZ98niV-KKVsEgl5LSavtTrwlkJZ"  # Replace with your Google Drive ID
    model_path = "yolov8.pt"
    if not os.path.exists(model_path):
        print("🔽 Downloading YOLOv8 model...")
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✅ Model download complete.")

# Step 2: Call the download function BEFORE loading the model
download_model()

# Step 3: Load the YOLO model
model = YOLO("yolov8.pt")

@app.route('/')
def home():
    return "Wildlife Monitoring YOLO API is running."

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join("temp", file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    detected_species = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        names = model.names
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                detected_species.append(names[cls])

    cap.release()
    os.remove(filepath)

    return jsonify({
        "species_detected": list(set(detected_species))
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
