# import cv2
# import numpy as np
# import os
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping

# # Cấu hình thư mục chứa dữ liệu
# data_dir = "data"  # Đảm bảo rằng thư mục "data" chứa các thư mục con (ví dụ, "person_1", "person_2", ...)

# # Đọc dữ liệu từ thư mục (cả hình ảnh và video)
# def load_data(data_dir, max_frames=10):
#     data = []
#     labels = []
    
#     for label in os.listdir(data_dir):
#         label_dir = os.path.join(data_dir, label)
        
#         if os.path.isdir(label_dir):  # Kiểm tra nếu là thư mục con (mỗi thư mục con là một nhãn)
#             for file_name in os.listdir(label_dir):
#                 file_path = os.path.join(label_dir, file_name)
                
#                 if file_name.lower().endswith(('jpg', 'jpeg', 'png')):  # Nếu là file ảnh
#                     img = cv2.imread(file_path)
#                     img_resized = cv2.resize(img, (64, 64))  # Resize ảnh về kích thước cố định
#                     data.append(img_resized)
#                     labels.append(label)
                
#                 elif file_name.lower().endswith(('mp4', 'avi', 'mov')):  # Nếu là file video
#                     cap = cv2.VideoCapture(file_path)
#                     frame_count = 0
                    
#                     while cap.isOpened():
#                         ret, frame = cap.read()
#                         if not ret or frame_count >= max_frames:
#                             break
#                         frame_resized = cv2.resize(frame, (64, 64))  # Resize khung hình
#                         data.append(frame_resized)
#                         labels.append(label)
#                         frame_count += 1
                    
#                     cap.release()
    
#     data = np.array(data)
#     labels = np.array(labels)

#     # Mã hóa nhãn
#     le = LabelEncoder()
#     le.fit(labels)
#     labels = le.transform(labels)
#     labels = to_categorical(labels)

#     return data, labels, le

# # Huấn luyện mô hình
# def train_model():
#     # Tải dữ liệu từ thư mục
#     data, labels, le = load_data(data_dir)

#     # Kiểm tra số lượng dữ liệu
#     print(f"Number of training images and frames: {len(data)}")

#     # Kiểm tra nếu chỉ có một mẫu
#     if len(data) == 1:
#         print("Warning: Only one sample found. Training will proceed with this single sample.")

#     model_path = 'models/face_recognition_model.h5'
#     if os.path.exists(model_path):
#         print("============= USING EXISTED MODEL TO TRAIN ====================")
#         model = load_model(model_path)
#     else:
#         # Xây dựng mô hình CNN
#         model = Sequential([
#             Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#             MaxPooling2D((2, 2)),
#             Conv2D(64, (3, 3), activation='relu'),
#             MaxPooling2D((2, 2)),
#             Conv2D(128, (3, 3), activation='relu'),
#             MaxPooling2D((2, 2)),
#             Flatten(),
#             Dropout(0.5),
#             Dense(128, activation='relu'),
#             Dense(len(le.classes_), activation='softmax')
#         ])

#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#     # Data augmentation
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )

#     # Áp dụng data augmentation
#     datagen.fit(data)

#     # Nếu chỉ có một mẫu, huấn luyện trực tiếp trên toàn bộ dữ liệu
#     if len(data) == 1:
#         history = model.fit(datagen.flow(data, labels, batch_size=1), epochs=50, verbose=1)
#     else:
#         # Chia dữ liệu thành train và validation nếu có nhiều mẫu
#         from sklearn.model_selection import train_test_split
#         X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
#         history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
#                             epochs=50,
#                             verbose=1,
#                             validation_data=(X_val, y_val),
#                             callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

#     # Kiểm tra lịch sử huấn luyện
#     print(f"Training history: {history.history}")

#     # Lưu mô hình
#     model.save('models/face_recognition_model.h5')
#     print("Model training complete!")

#     # Lưu label encoder
#     np.save('models/label_encoder.npy', le.classes_)

# # Chạy huấn luyện mô hình
# train_model()

# ============================================================================================
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import keras

# Cấu hình thư mục chứa dữ liệu
data_dir = "data"  # Đảm bảo rằng thư mục "data" chứa các thư mục con (ví dụ, "person_1", "person_2", ...)

# Đọc dữ liệu từ thư mục (cả hình ảnh và video)
def load_data(data_dir, max_frames=100):
    data = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        
        if os.path.isdir(label_dir):  # Kiểm tra nếu là thư mục con (mỗi thư mục con là một nhãn)
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                
                if file_name.lower().endswith(('jpg', 'jpeg', 'png')):  # Nếu là file ảnh
                    img = cv2.imread(file_path)
                    img_resized = cv2.resize(img, (64, 64))  # Resize ảnh về kích thước cố định
                    data.append(img_resized)
                    labels.append(label)
                
                elif file_name.lower().endswith(('mp4', 'avi', 'mov')):  # Nếu là file video
                    cap = cv2.VideoCapture(file_path)
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame_count >= max_frames:
                            break
                        frame_resized = cv2.resize(frame, (64, 64))  # Resize khung hình
                        data.append(frame_resized)
                        labels.append(label)
                        frame_count += 1
                    
                    cap.release()
    
    data = np.array(data)
    labels = np.array(labels)

    # Mã hóa nhãn
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    labels = to_categorical(labels)

    return data, labels, le

# Huấn luyện mô hình
def train_model():
    # Tải dữ liệu từ thư mục
    data, labels, le = load_data(data_dir)

    # Kiểm tra số lượng dữ liệu
    print(f"Number of training images and frames: {len(data)}")

    # Kiểm tra nếu chỉ có một mẫu
    if len(data) == 1:
        print("Warning: Only one sample found. Training will proceed with this single sample.")

    model_path = 'models/my_model2.h5'
    
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes (output units): {labels.shape[1]}")


    if os.path.exists(model_path):
        print("============= USING EXISTED MODEL TO TRAIN ====================")
        model = load_model(model_path)
    else:
        # Xây dựng mô hình CNN
        num_classes = labels.shape[1]
        
        model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(64, 64, 3)),
            keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),

            keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
            keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),

            keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
            keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),

            keras.layers.Flatten(),
            
            keras.layers.Dense(units=512, activation='relu'),  # Adjust based on Flatten output
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=num_classes, activation='softmax')  # Match the number of classes
        ])



    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Áp dụng data augmentation
    datagen.fit(data)

    # Nếu chỉ có một mẫu, huấn luyện trực tiếp trên toàn bộ dữ liệu
    if len(data) == 1:
        history = model.fit(datagen.flow(data, labels, batch_size=1), epochs=50, verbose=1)
    else:
        # Chia dữ liệu thành train và validation nếu có nhiều mẫu
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
        history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                            epochs=5,
                            verbose=1,
                            validation_data=(X_val, y_val),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

    # Kiểm tra lịch sử huấn luyện
    print(f"Training history: {history.history}")

    # Lưu mô hình
    model.save('models/my_model2.h5')
    print("Model training complete!")

    # Lưu label encoder
    np.save('models/label_encoder.npy', le.classes_)

# Chạy huấn luyện mô hình
train_model()

#==================================================================================
# import cv2
# import numpy as np
# import os
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# import keras

# # Load OpenCV's pre-trained model for face landmark detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# landmark_model = cv2.face.createFacemarkLBF()
# landmark_model.loadModel('lbfmodel.yaml')  # Download this file from OpenCV's GitHub repo

# # Hàm để phát hiện mặt và trích xuất keypoints
# def extract_keypoints(face_image):
#     gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     keypoints = []

#     for (x, y, w, h) in faces:
#         face_region = face_image[y:y+h, x:x+w]
        
#         # Detect landmarks/keypoints for each detected face
#         success, landmarks = landmark_model.fit(face_region)
        
#         if success:
#             for landmark in landmarks[0]:
#                 keypoints.append(landmark[0])  # Append each (x, y) coordinate of the keypoints
#     return np.array(keypoints)

# # Đọc dữ liệu từ thư mục (cả hình ảnh và video)
# def load_data(data_dir, max_frames=10):
#     data = []
#     labels = []
#     keypoints_data = []
    
#     for label in os.listdir(data_dir):
#         label_dir = os.path.join(data_dir, label)
        
#         if os.path.isdir(label_dir):  # Kiểm tra nếu là thư mục con (mỗi thư mục con là một nhãn)
#             for file_name in os.listdir(label_dir):
#                 file_path = os.path.join(label_dir, file_name)
                
#                 if file_name.lower().endswith(('jpg', 'jpeg', 'png')):  # Nếu là file ảnh
#                     img = cv2.imread(file_path)
#                     img_resized = cv2.resize(img, (64, 64))  # Resize ảnh về kích thước cố định
#                     keypoints = extract_keypoints(img)
                    
#                     if keypoints is not None:
#                         data.append(img_resized)
#                         labels.append(label)
#                         keypoints_data.append(keypoints.flatten())  # Flatten keypoints to use as a vector
                        
#                 elif file_name.lower().endswith(('mp4', 'avi', 'mov')):  # Nếu là file video
#                     cap = cv2.VideoCapture(file_path)
#                     frame_count = 0
                    
#                     while cap.isOpened():
#                         ret, frame = cap.read()
#                         if not ret or frame_count >= max_frames:
#                             break
#                         frame_resized = cv2.resize(frame, (64, 64))  # Resize khung hình
#                         keypoints = extract_keypoints(frame)
                        
#                         if keypoints is not None:
#                             data.append(frame_resized)
#                             labels.append(label)
#                             keypoints_data.append(keypoints.flatten())  # Flatten keypoints to use as a vector
#                         frame_count += 1
#                     cap.release()

#     data = np.array(data)
#     labels = np.array(labels)
#     keypoints_data = np.array(keypoints_data)

#     # Mã hóa nhãn
#     le = LabelEncoder()
#     le.fit(labels)
#     labels = le.transform(labels)
#     labels = to_categorical(labels)

#     return data, labels, keypoints_data, le

# # Huấn luyện mô hình
# def train_model():
#     # Tải dữ liệu từ thư mục
#     data, labels, keypoints_data, le = load_data(data_dir)

#     # Kiểm tra số lượng dữ liệu
#     print(f"Number of training images and frames: {len(data)}")

#     # Kiểm tra nếu chỉ có một mẫu
#     if len(data) == 1:
#         print("Warning: Only one sample found. Training will proceed with this single sample.")

#     model_path = 'models/face_recognition_model2.keras'
#     if os.path.exists(model_path):
#         print("============= USING EXISTED MODEL TO TRAIN ====================")
#         model = load_model(model_path)
#     else:
#         # Xây dựng mô hình CNN kết hợp với keypoints
#         image_input = Input(shape=(64, 64, 3))
#         x = Conv2D(filters=64, kernel_size=3, activation='relu')(image_input)
#         x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
#         x = MaxPooling2D(pool_size=2, strides=2)(x)

#         x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
#         x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
#         x = MaxPooling2D(pool_size=2, strides=2)(x)

#         x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
#         x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
#         x = MaxPooling2D(pool_size=2, strides=2)(x)

#         x = Flatten()(x)

#         # Tích hợp thông tin keypoints
#         keypoints_input = Input(shape=(136,))  # 68 keypoints * 2 (x, y)
#         keypoints_x = Dense(128, activation='relu')(keypoints_input)
#         keypoints_x = Dropout(0.5)(keypoints_x)

#         # Kết hợp hai nguồn dữ liệu: ảnh và keypoints
#         combined = concatenate([x, keypoints_x])

#         # Lớp Dense cuối cùng
#         final_output = Dense(units=labels.shape[1], activation='softmax')(combined)

#         model = keras.Model(inputs=[image_input, keypoints_input], outputs=final_output)

#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#     # Data augmentation
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )

#     # Áp dụng data augmentation
#     datagen.fit(data)

#     # Nếu chỉ có một mẫu, huấn luyện trực tiếp trên toàn bộ dữ liệu
#     if len(data) == 1:
#         history = model.fit([data, keypoints_data], labels, batch_size=1, epochs=50, verbose=1)
#     else:
#         # Chia dữ liệu thành train và validation nếu có nhiều mẫu
#         from sklearn.model_selection import train_test_split
#         X_train, X_val, y_train, y_val, keypoints_train, keypoints_val = train_test_split(
#             data, labels, keypoints_data, test_size=0.2, random_state=42
#         )
#         history = model.fit(
#             [X_train, keypoints_train], y_train, batch_size=32, epochs=50, verbose=1,
#             validation_data=([X_val, keypoints_val], y_val),
#             callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
#         )

#     # Kiểm tra lịch sử huấn luyện
#     print(f"Training history: {history.history}")

#     # Lưu mô hình
#     model.save('models/my_model_with_keypoints.h5')
#     print("Model training complete!")

#     # Lưu label encoder
#     np.save('models/label_encoder.npy', le.classes_)

# # Chạy huấn luyện mô hình
# train_model()