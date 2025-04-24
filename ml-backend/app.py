from flask import Flask, request, jsonify, render_template
from utils import process_image, predict_with_vgg16  # Import functions from utils.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(img_path)
        try:
            predictions = process_image(img_path)
        except ImportError as e:
            return jsonify({'error': str(e) + ' Please install or clone yolov5 repository properly.'}), 500
        except Exception as e:
            return jsonify({'error': 'Processing error: ' + str(e)}), 500
        return jsonify({'predictions': predictions})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/vgg16', methods=['POST'])
def vgg16():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # Check file extension for supported formats (exclude avif)
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext == 'avif':
            return jsonify({'error': 'AVIF image format not supported. Please upload JPEG or PNG images.'}), 400
        filename = secure_filename(file.filename)
        img_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(img_path)
        try:
            predictions = predict_with_vgg16(img_path)
        except Exception as e:
            return jsonify({'error': 'Processing error: ' + str(e)}), 500
        return jsonify({'predictions': predictions})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)

