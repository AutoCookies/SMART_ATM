import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2, l1
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from tensorflow.keras.models import load_model


# Constants
DATA_DIR = "data\\converted_data2"  # Updated directory containing converted .wav files
MODEL_DIRECTORY = "models\\speech_to_text"
MAX_LEN = 30  # Adjusted MAX_LEN to match audio lengths
N_MFCC = 13    # Number of MFCC features

# Constants for callbacks
PATIENCE_LR = 3  # Number of epochs with no improvement to reduce learning rate
FACTOR_LR = 0.5  # Factor by which the learning rate will be reduced
MIN_LR = 1e-8    # Minimum learning rate
PATIENCE_ES = 6  # Number of epochs with no improvement to stop training
EPOCHS = 100 # Number of times the model trains

def compress_audio(audio, threshold=0.8, ratio=1.25):
    """
    Apply time stretching to compress the dynamic range of the audio.
    Parameters:
        audio (numpy.ndarray): Input audio signal.
        threshold (float): Not used directly, can be removed or adjusted for other compression techniques.
        ratio (float): Time-stretching factor; >1 compresses (speeds up), <1 expands (slows down).
    Returns:
        numpy.ndarray: Compressed audio signal.
    """
    audio = librosa.effects.preemphasis(audio)  # Optional, may help in some cases
    compressed_audio = librosa.effects.time_stretch(audio, rate=ratio)  # Corrected function call
    return np.clip(compressed_audio, -1, 1)  # Ensure audio remains in the [-1, 1] range

def normalize_audio(audio):
    # Normalize the audio between -1 and 1
    audio_max = np.max(np.abs(audio))
    if audio_max == 0:
        print("Warning: Audio signal is silent. Skipping normalization.")
        return audio  # Return original audio if it is silent
    return audio / audio_max


def augment_audio(audio, sample_rate, noise_factor=0.005, stretch_rate=1.1, pitch_steps=4):
    # Adding noise
    noise = np.random.randn(len(audio))
    audio_noisy = audio + noise_factor * noise

    # Time stretching
    audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)

    # Pitch shifting
    audio_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_steps)

    return audio_noisy, audio_stretched, audio_shifted

def pad_audio_features(mfcc, max_len=MAX_LEN):
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_log_mel_spectrogram(audio, sample_rate=16000, n_mels=40, fmin=20, fmax=8000):
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmin=fmin, fmax=fmax)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def normalize_mfcc(mfcc):
    # Normalize MFCC features to have zero mean and unit variance
    return (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)


def load_data(data_dir, max_len=MAX_LEN, max_files_per_label=50):
    features, labels, texts = [], [], []
    label_map = {}

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")
    
    for idx, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label_map[folder] = idx
            file_count = 0

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".wav") and file_count < max_files_per_label:
                    file_path = os.path.join(folder_path, file_name)

                    transcription_path = os.path.join(folder_path, "transcription.txt")
                    if not os.path.exists(transcription_path):
                        print(f"No transcription found for {folder_path}")
                        continue

                    with open(transcription_path, "r") as f:
                        transcription = f.read().strip()

                    if len(transcription) == 0:
                        continue

                    audio, sample_rate = librosa.load(file_path, sr=16000)

                    # Apply normalization and compression
                    audio = normalize_audio(audio)
                    audio = compress_audio(audio)

                    # Apply augmentation
                    audio_noisy, audio_stretched, audio_shifted = augment_audio(audio, sample_rate)

                    # Extract MFCCs for the original and augmented versions
                    for augmented_audio in [audio, audio_noisy, audio_stretched, audio_shifted]:
                        mfcc = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=N_MFCC)
                        mfcc = normalize_mfcc(mfcc)
                        mfcc = pad_audio_features(mfcc, max_len)

                        features.append(mfcc.T)
                        labels.append(idx)
                        texts.append(transcription)

                    file_count += 1

    print(f"Loaded {len(features)} audio samples.")
    print(f"Loaded {len(texts)} transcriptions.")

    if len(texts) == 0:
        raise ValueError("No valid transcriptions found.")

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1

    texts = tokenizer.texts_to_sequences(texts)
    texts = tf.keras.preprocessing.sequence.pad_sequences(texts, padding='post')

    return np.array(features), np.array(texts), vocab_size

def build_speech_to_text_model(input_shape, vocab_size):
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=3)(inputs)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    print("Loading data...")
    features, texts, vocab_size = load_data(DATA_DIR)

    print(f"Data loaded with {features.shape[0]} samples, {vocab_size} unique tokens.")

    if len(texts) == 0:
        raise ValueError("No text data available to train.")

    X_train, X_test, y_train, y_test = train_test_split(features, texts, test_size=0.3)

    input_shape = X_train.shape[1:]

    model = build_speech_to_text_model(input_shape, vocab_size)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")

    early_stopping = EarlyStopping(monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=32, callbacks=[early_stopping])

    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    model.save(os.path.join(MODEL_DIRECTORY, "speech_to_text_model.h5"))

    print(f"Model saved to {os.path.join(MODEL_DIRECTORY, 'speech_to_text_model.h5')}")
