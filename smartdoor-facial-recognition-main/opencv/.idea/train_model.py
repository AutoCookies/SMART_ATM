import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from pymongo import MongoClient
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import base64
import os
from flask import Flask
from tensorflow.keras.mixed_precision import set_global_policy

# Cài đặt Mixed Precision để tăng tốc độ huấn luyện
set_global_policy('mixed_float16')

app = Flask(__name__)

# Hàm giải mã ảnh từ base64
def decode_base64_image(base64_bytes):
    base64_string = base64_bytes.decode('utf-8')
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img_np
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# Hàm để load hoặc tạo mới mô hình
def load_or_create_model(num_classes):
    if os.path.exists('models/model.h5'):
        print("Loading existing model...")
        model = load_model('models/model.h5')
        
        # Kiểm tra số lớp đầu ra của mô hình và số lớp từ nhãn
        if model.output_shape[-1] != num_classes:
            print(f"Model output shape {model.output_shape[-1]} does not match number of classes {num_classes}. Recreating model...")
            # Tạo lại mô hình nếu số lớp không khớp
            model = create_new_model(num_classes)
    else:
        print("Creating new model...")
        model = create_new_model(num_classes)

    return model

# Hàm tạo mới mô hình với số lớp đầu ra chính xác
def create_new_model(num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Giảm số lượng filter để tăng tốc độ
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),  # Thêm lớp Convolution để tăng khả năng học
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),  # Giảm số lượng neurons trong lớp Dense
        Dense(num_classes, activation='softmax', dtype='float32')  # Đảm bảo lớp cuối cùng là float32 và số lớp chính xác
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hàm để load hoặc tạo mới LabelEncoder
def load_or_create_label_encoder():
    if os.path.exists('models/label_encoder.npy'):
        print("Loading existing LabelEncoder...")
        classes = np.load('models/label_encoder.npy', allow_pickle=True)
        le = LabelEncoder()
        le.classes_ = classes
    else:
        le = LabelEncoder()
    return le

# Hàm huấn luyện mô hình
def train():
    client = MongoClient("mongodb+srv://admin:admin@cluster0.bax6c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["test"]
    collection = db["users"]

    data = []
    labels = []

    le = load_or_create_label_encoder()
    existing_classes = set(le.classes_) if len(le.classes_) > 0 else set()

    for doc in collection.find():
        username = doc.get("username", "unknown")
        images = doc.get("images", [])

        for idx, image_base64 in enumerate(images):
            img_np = decode_base64_image(image_base64)

            if img_np is not None:
                print(f"Successfully loaded image {idx} for user {username}")
                data.append(cv2.resize(img_np, (64, 64)))  # Resize ảnh về 64x64
                labels.append(username)
            else:
                print(f"Failed to decode image {idx} for user {username}")

    if not data:
        print("No valid data found. Training aborted.")
        return

    data = np.array(data)
    print(f"Number of training images: {len(data)}")
    print(f"Labels used: {set(labels)}")

    le.fit(labels + list(existing_classes))
    labels = le.transform(labels)
    labels = to_categorical(labels)

    # Đảm bảo số lượng lớp đầu ra khớp với số lượng lớp của nhãn
    model = load_or_create_model(len(le.classes_))
    print("Starting training with new data...")
    model.fit(data, labels, epochs=5, batch_size=8, verbose=1)

    model.save('models/model.h5')
    np.save('models/label_encoder.npy', le.classes_)
    print("Model trained and updated successfully!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
