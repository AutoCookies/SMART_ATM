import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Constants
MODEL_DIRECTORY = "voice_model"
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
# # import os
# # import numpy as np
# # import librosa
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # import tkinter as tk
# # from tkinter import filedialog  # For file dialog to select audio file
# # import sounddevice as sd  # For recording audio
# # import scipy.signal

# # # Constants
# # MODEL_DIRECTORY = "voice_model"
# # MODEL_PATH = os.path.join(MODEL_DIRECTORY, "rnn_attention_model.h5")
# # LABEL_MAP_PATH = os.path.join(MODEL_DIRECTORY, "label_map.txt")
# # SAMPLERATE = 16000  # Sample rate for audio recording
# # N_MFCC = 13         # Number of MFCC features
# # MAX_LEN = 30         # Maximum length of the MFCC

# # # Function to load the label map
# # def load_label_map(label_map_path):
# #     label_map = {}
# #     with open(label_map_path, "r") as f:
# #         for line in f:
# #             label, idx = line.strip().split("\t")
# #             label_map[int(idx)] = label
# #     return label_map

# # # Function to preprocess the audio and extract MFCC features
# # def preprocess_audio(audio, sample_rate=SAMPLERATE, max_len=MAX_LEN, n_mfcc=N_MFCC):
# #     mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
# #     if mfcc.shape[1] < max_len:
# #         pad_width = max_len - mfcc.shape[1]
# #         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
# #     else:
# #         mfcc = mfcc[:, :max_len]
# #     return mfcc.T  # Transpose to match the input shape

# # # Function to record audio from the microphone
# # def record_audio(duration=3, samplerate=SAMPLERATE):
# #     print("Recording...")
# #     audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
# #     sd.wait()  # Wait until the recording is finished
# #     return audio.flatten()

# # # Function to upload an audio file using a file dialog
# # def upload_audio_file():
# #     root = tk.Tk()
# #     root.withdraw()  # Hide the root window
# #     file_path = filedialog.askopenfilename(title="Select an audio file", filetypes=[("WAV files", "*.wav")])
# #     return file_path

# # # Main testing script
# # if __name__ == "__main__":
# #     # Load the model and label map
# #     print("Loading model...")
# #     model = load_model(MODEL_PATH)
# #     label_map = load_label_map(LABEL_MAP_PATH)
# #     print("Model and label map loaded.")

# #     # Upload an audio file
# #     audio_file_path = upload_audio_file()
# #     if not audio_file_path:
# #         print("No file selected. Exiting.")
# #         exit()

# #     # Load the selected audio file
# #     audio, sample_rate = librosa.load(audio_file_path, sr=SAMPLERATE)

# #     # Preprocess the audio
# #     audio_mfcc = preprocess_audio(audio, sample_rate=sample_rate)

# #     # Expand dimensions to match model input (batch size, time steps, features)
# #     audio_mfcc = np.expand_dims(audio_mfcc, axis=0)

# #     # Make prediction
# #     print("Predicting...")
# #     prediction = model.predict(audio_mfcc)
# #     predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability

# #     # Map the predicted class index to the actual label
# #     predicted_label = label_map.get(predicted_class, "Unknown")

# #     # Output the result
# #     print(f"Predicted class index: {predicted_class}")
# #     print(f"Predicted label: {predicted_label}")

