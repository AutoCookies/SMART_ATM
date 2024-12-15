import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

# Hướng dẫn folder chứa dữ liệu
data_dir = 'clone_model\\opencv\\.idea\\data\\sign_folder'

# Sử dụng ImageDataGenerator để load dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir),
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(data_dir),
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load mô hình VGG16 đã huấn luyện sẵn, bỏ qua phần Fully Connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Đặt base_model không huấn luyện
base_model.trainable = False

# Xây dựng mô hình với base_model đã học sẵn
model = Sequential([
    base_model,  # VGG16 pretrained model
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Assuming 2 classes (e.g., "S" and "O")
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Định nghĩa các callbacks
earlyStopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model_vgg16.keras', save_best_only=True)

# Huấn luyện mô hình
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[earlyStopping, model_checkpoint]
)

# Save mô hình cuối cùng
model.save('signLang_final_model_vgg16.keras')
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
