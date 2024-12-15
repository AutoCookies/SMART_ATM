import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Constants
MODEL_DIRECTORY = "modes/voice_model"
N_MFCC = 13  # Number of MFCC features
MAX_LEN = 30  # Adjusted MAX_LEN to match audio lengths
SAMPLE_RATE = 16000  # Sample rate for audio recording
AUDIO_DURATION = 3  # Duration of the audio in seconds

# Load the model
model = load_model(os.path.join(MODEL_DIRECTORY, "rnn_attention_model.h5"))

# Load the label map
label_map = {}
with open(os.path.join(MODEL_DIRECTORY, "label_map.txt"), "r") as f:
    for line in f:
        label, idx = line.strip().split("\t")
        label_map[int(idx)] = label

# Function to capture audio from the microphone
def record_audio(duration=AUDIO_DURATION, sample_rate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    print("Recording complete")
    return audio.flatten()

# Function to process the audio and extract MFCC features
def extract_mfcc_from_audio(audio, sample_rate=SAMPLE_RATE):
    audio = librosa.util.normalize(audio)  # Normalize audio
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)  # Normalize MFCC
    # Padding or truncating to ensure fixed length
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc.T  # Transpose to match the expected input shape

# Function to predict the class of the audio input
def predict_audio_class(audio):
    mfcc = extract_mfcc_from_audio(audio)  # Extract MFCC features
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return label_map[predicted_class]

# Main loop for real-time prediction
if __name__ == "__main__":
    while True:
        # Record audio from the microphone
        audio = record_audio(duration=AUDIO_DURATION)

        # Predict the label of the recorded audio
        predicted_label = predict_audio_class(audio)
        
        # Print the predicted label
        print(f"Predicted label: {predicted_label}")
        
        # Wait a bit before recording again
        time.sleep(1)  # Adjust the delay if necessary

# # ====================== UPLOAD VERSION ==============================
# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import tkinter as tk
# from tkinter import filedialog, messagebox

# # Constants
# MODEL_PATH = "models\\voice_model\\voice_attention_test_reg_4.h5"
# LABEL_MAP_PATH = "models\\voice_model\\label_map.txt"
# SAMPLERATE = 16000  # Tốc độ lấy mẫu cho ghi âm/audio
# N_MFCC = 13         # Số đặc trưng MFCC
# MAX_LEN = 30        # Độ dài tối đa MFCC

# # Load model và label map
# def load_label_map(label_map_path):
#     label_map = {}
#     with open(label_map_path, "r") as f:
#         for line in f:
#             label, idx = line.strip().split("\t")
#             label_map[int(idx)] = label
#     return label_map

# # Preprocess audio để tạo MFCC
# def preprocess_audio(audio, sample_rate=SAMPLERATE, max_len=MAX_LEN, n_mfcc=N_MFCC):
#     mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
#     if mfcc.shape[1] < max_len:
#         pad_width = max_len - mfcc.shape[1]
#         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfcc = mfcc[:, :max_len]
#     return mfcc.T

# # Dự đoán nhãn từ tệp âm thanh
# def predict_label(file_path):
#     audio, sample_rate = librosa.load(file_path, sr=SAMPLERATE)
#     audio_mfcc = preprocess_audio(audio, sample_rate)

#     audio_mfcc = np.expand_dims(audio_mfcc, axis=0)  # Batch size, time steps, features
#     prediction = model.predict(audio_mfcc)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     predicted_label = label_map.get(predicted_class, "Unknown")
    
#     return predicted_label

# # Chọn tệp âm thanh từ giao diện
# def select_audio_file():
#     file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
#     if file_path:
#         label = predict_label(file_path)
#         messagebox.showinfo("Prediction Result", f"Predicted Label: {label}")

# # Tạo giao diện Tkinter
# root = tk.Tk()
# root.title("Voice Prediction GUI")

# model = load_model(MODEL_PATH)
# label_map = load_label_map(LABEL_MAP_PATH)

# tk.Label(root, text="Click button to select audio file").pack(pady=10)
# btn_select = tk.Button(root, text="Select File", command=select_audio_file)
# btn_select.pack(pady=20)

# # Chạy giao diện Tkinter
# root.mainloop()