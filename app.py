from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image

model1 = YOLO(r"/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/Models/yolo_v8/Medium/weights/best.pt")
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
        return jsonify({'error': 'No image file found in the request'}), 400
    
    file = request.files['file']
    model_choice = request.form.get('model')
    
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

    # YOLO Prediction
    results = model.predict(img, conf=0.2)
    img_with_boxes = results[0].plot()  # Matplotlib array

    # Convert Matplotlib image to send via Flask
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_with_boxes)
    ax.axis('off')
    canvas = FigureCanvas(fig)
    img_io = io.BytesIO()
    canvas.print_png(img_io)
    img_io.seek(0)

    # Convert the image to RGB (to avoid RGBA error when saving as JPEG)
    img_pil = Image.open(img_io).convert('RGB')

    # Save the processed image in the folder
    processed_filepath = os.path.join(PROCESSED_FOLDER, filename)
    img_pil.save(processed_filepath, format='JPEG')  # Save as JPEG

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
