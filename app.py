from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os

model1 = YOLO("/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/best.pt")
model2 = ""
model = ""
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        print("Error: No image file found in the request")
        return jsonify({'error': 'No image file found in the request'}), 400
    
    file = request.files['file']
    print("File received:", file)
    
    model_choice = request.form.get('model')  # Get the model choice from the form
    print("Model selected:", model_choice)

    if model_choice == "model1":
        model = model1
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Process the image with the model
    results = model(img)
    predictions = results[0].boxes.data.tolist()

    # Draw bounding boxes on the image
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    processed_filepath = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(processed_filepath, img)

    return jsonify({
        'aiResponse': 'Prediction successful',
        'image_url': f'/static/processed/{filename}',
        'helpline': 'Contact us at 1-800-CAR-HELP'
    })


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
