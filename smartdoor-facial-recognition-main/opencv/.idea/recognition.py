# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# app = Flask(__name__)
# CORS(app)  # Kích hoạt CORS

# # Load mô hình và label encoder
# model = load_model('models/model.h5')
# classes = np.load('models/label_encoder.npy', allow_pickle=True)

# def recognize_image(image):
#     npimg = np.frombuffer(image, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     resized_frame = cv2.resize(img, (64, 64))
#     resized_frame = np.expand_dims(resized_frame, axis=0)

#     prediction = model.predict(resized_frame)
#     label_index = np.argmax(prediction)

#     if prediction[0][label_index] < 0.6:
#         return "Unknown"

#     label = classes[label_index]
#     return label
# @app.route('/recognize', methods=['OPTIONS'])
# def handle_options():
#     response = jsonify({'message': 'CORS preflight passed'})
#     response.headers.add("Access-Control-Allow-Origin", "*")
#     response.headers.add("Access-Control-Allow-Headers", "Content-Type")
#     response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
#     return response




# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5050, debug=True)
