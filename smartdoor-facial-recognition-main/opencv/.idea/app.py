from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS  # Import Flask-CORS
import data_collection
import train_model
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64  # Thêm dòng này

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS cho toàn bộ ứng dụng


def recognize_image(image):
    npimg = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    resized_frame = cv2.resize(img, (64, 64))
    resized_frame = np.expand_dims(resized_frame, axis=0)

    prediction = model.predict(resized_frame)
    label_index = np.argmax(prediction)

    if prediction[0][label_index] < 0.6:
        return "Unknown"

    label = classes[label_index]
    return label

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    data_collection.collect_data(request)
    return redirect(url_for('home'))

@app.route('/train_model')
def train():
    train_model.train()
    return redirect(url_for('home'))

@app.route('/recognize', methods=['POST'])
def recognize():
    # Load mô hình và label encoder
    model = load_model('models/model.h5')
    classes = np.load('models/label_encoder.npy', allow_pickle=True)

    print("POST request received")  # Log để kiểm tra xem có nhận yêu cầu POST không
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"label": None, "message": "No image uploaded"}), 400
        base64_string = data['image']
        header, encoded = base64_string.split(',', 1)
        image = base64.b64decode(encoded)  # Đây sẽ sử dụng thư viện base64
        label = recognize_image(image)
        return jsonify({"label": label})

    except Exception as e:
        print(f"Error processing request: {str(e)}")  # In lỗi ra console
        return jsonify({"error": "An error occurred", "message": str(e)}), 500

# Thêm route xử lý OPTIONS nếu cần
@app.route('/recognize', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'message': 'CORS preflight passed'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)  # Chạy server trên cổng 5050
