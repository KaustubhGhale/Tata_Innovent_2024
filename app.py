from flask import Flask, render_template, request, jsonify
'''from models.model1 import detect_dents
from models.model2 import detect_scratches'''

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

'''@app.route('/process', methods=['POST'])
def process_file():
    file = request.files['file']
    model = request.form['model']
    
    # Example processing based on model selected
    if model == 'model1':
        result = detect_dents(file)
    elif model == 'model2':
        result = detect_scratches(file)
    # Add more models as needed
    
    # Generate AI response and repair suggestions
    ai_response = generate_ai_response(result)
    repaired_image_path = generate_repaired_image(result)
    
    return jsonify({
        'aiResponse': ai_response,
        'generatedImage': repaired_image_path,
        'helpline': "Call 1800-XYZ-CARS for assistance."
    })
'''
if __name__ == '__main__':
    app.run(debug=True)
