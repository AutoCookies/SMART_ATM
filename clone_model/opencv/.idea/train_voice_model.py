import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense, Conv1D, GlobalAveragePooling1D, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2, l1, l1_l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from tensorflow.keras.models import load_model


# Constants
DATA_DIR = "data\\converted_data2"  # Updated directory containing converted .wav files
MODEL_NAME = "voice_attention_test_reg_5.h5"
MODEL_DIRECTORY = "models\\voice_model"
MAX_LEN = 30  # Adjusted MAX_LEN to match audio lengths
N_MFCC = 13    # Number of MFCC features

# Constants for callbacks
PATIENCE_LR = 3  # Number of epochs with no improvement to reduce learning rate
FACTOR_LR = 0.5  # Factor by which the learning rate will be reduced
MIN_LR = 1e-8    # Minimum learning rate
PATIENCE_ES = 6  # Number of epochs with no improvement to stop training
EPOCHS = 100 # Number of times the model trains

def compress_audio(audio, threshold=0.8, ratio=0.85):
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


def augment_audio(audio, sample_rate, noise_factor=0.007, stretch_rate=1.3, pitch_steps=5):
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

def add_noise(audio, noise, snr_db=10):
    """
    Thêm tiếng ồn vào audio với mức SNR (Signal-to-Noise Ratio) cụ thể.
    :param audio: Tín hiệu âm thanh gốc.
    :param noise: Tín hiệu tiếng ồn.
    :param snr_db: Tỷ lệ tín hiệu trên tiếng ồn (dB), phải nằm trong khoảng hợp lý.
    :return: Tín hiệu âm thanh đã thêm tiếng ồn.
    """
    # Giới hạn snr_db
    if snr_db > 50 or snr_db < 0:  # Giới hạn SNR hợp lý từ 0 đến 50 dB
        raise ValueError("snr_db must be between 0 and 50 dB.")

    # Điều chỉnh độ dài của noise để khớp với audio
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))[:len(audio)]
    else:
        noise = noise[:len(audio)]
    
    # Tính toán công suất tín hiệu
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Tính toán tỷ lệ SNR mong muốn
    try:
        target_noise_power = audio_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)
    except OverflowError as e:
        raise ValueError(f"Calculation overflow: SNR value ({snr_db}) is too high.") from e
    
    # Hòa trộn
    audio_noisy = audio + noise
    return audio_noisy


def load_data(data_dir, max_len=MAX_LEN, max_files_per_label=50):
    features, labels = [], []
    label_map = {}

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")
    
    noise, _ = librosa.load("stuff_to_enhance_model/crowd-worried-90368.wav", sr=16000)
    
    for idx, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label_map[folder] = idx
            file_count = 0  # Track number of files processed for each label
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".wav") and file_count < max_files_per_label:
                    file_path = os.path.join(folder_path, file_name)
                    audio, sample_rate = librosa.load(file_path, sr=16000)

                    # Apply normalization and compression
                    audio = normalize_audio(audio)
                    audio = compress_audio(audio)
                    # noisy_audio = add_noise(audio, noise, 50)
                    # Apply augmentation
                    audio_noisy, audio_stretched, audio_shifted = augment_audio(audio, sample_rate)

                    # Extract MFCCs for the original and augmented versions
                    for augmented_audio in [audio, audio_noisy, audio_stretched, audio_shifted]:
                        mfcc = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=N_MFCC)
                        mfcc = normalize_mfcc(mfcc)  # Normalize MFCCs
                        mfcc = pad_audio_features(mfcc, max_len)
                        features.append(mfcc.T)
                        labels.append(idx)
                    
                    
                    # noisy_audio = add_noise(audio, noise, 50)
                    
                    # mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
                    # mfcc = pad_audio_features(mfcc, max_len)
                    # features.append(mfcc.T)
                    # labels.append(idx)

                    file_count += 1  # Increment file count for the current label

    print(f"Extracted {len(features)} samples.")
    return np.array(features), np.array(labels), label_map

def build_model_with_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Conv1D Block
    x = Conv1D(filters=64, kernel_size=3, kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    # LSTM Blocks
    x = LSTM(128, return_sequences=True, kernel_regularizer=l1(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(128, return_sequences=True, kernel_regularizer=l1(0.001))(x)
    x = BatchNormalization()(x)

    # Attention Layer
    attention = LayerNormalization(epsilon=1e-6)(attention)
    x = Dropout(0.3)(attention)

    # Global Pooling and Dense Layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, kernel_regularizer=l1(0.001))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_model_test_with_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Conv1D Block
    x = Conv1D(filters=64, kernel_size=3, kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # LSTM Block with residual connection
    x_residual = Conv1D(filters=128, kernel_size=1, padding="same")(x)  
    x = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_residual])

    # Multi-Head Self Attention Block
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64, attention_axes=[1])(x, x)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    x = Add()([x, attention_output])

    # Pre-Trained Sentence Embedding (Semantic Information)
    semantic_output = Dense(128, activation='relu')(x)  # Tạo thêm semantic features
    x = Add()([x, semantic_output])

    # Dense Layers with Regularization
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Dynamic learning rate scheduler
def get_dynamic_learning_rate_schedule(initial_lr=0.0001, total_steps=10000):
    # Choose between ExponentialDecay or CosineDecay
    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps // 10,
        decay_rate=0.9,
        staircase=True
    )
    return lr_schedule

# Plot accuracy and loss for training and validation
def plot_training_history(history, save_dir):
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(save_dir, "accuracy_plot_voice2.png")
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy plot saved to {accuracy_plot_path}")
    plt.show()
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_path = os.path.join(save_dir, "loss_plot_voice2.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    plt.show()
    plt.close()
    
# Main script for training the model
if __name__ == "__main__":
    print("Loading data...")
    features, labels, label_map = load_data(DATA_DIR)
    print(f"Loaded {features.shape[0]} samples with {len(label_map)} classes.")

    features = np.array(features)
    labels = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_map)

    # Check if model exists in the specified directory
    if os.path.exists(os.path.join(MODEL_DIRECTORY, MODEL_NAME)):
        print("Loading existing model...")
        model = load_model(os.path.join(MODEL_DIRECTORY, MODEL_NAME))
    else:
        print("Building new model with attention and regularization...")
        model = build_model_test_with_attention(input_shape, num_classes)

    # Dynamic Learning Rate Scheduler
    lr_schedule = get_dynamic_learning_rate_schedule(initial_lr=0.0001, total_steps=X_train.shape[0] * EPOCHS // 128)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_ES,
        restore_best_weights=True,
        verbose=1
    )

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=128,
        callbacks=[early_stopping],
        shuffle=True
    )

    # Save Model and Label Map
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    model.save(os.path.join(MODEL_DIRECTORY, MODEL_NAME))
    with open(os.path.join(MODEL_DIRECTORY, "label_map2.txt"), "w") as f:
        for label, idx in label_map.items():
            f.write(f"{label}\t{idx}\n")
    print("Model and label map saved.")

    # Evaluate the model on the test set
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128, verbose=1)

    # Print the evaluation results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    
    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get the index of the highest probability for each prediction

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Save the plot as an image
    confusion_matrix_path = os.path.join(MODEL_DIRECTORY, "confusion_matrix_voice2.png")
    plt.savefig(confusion_matrix_path)
    plt.close()  # Close the plot to free memory

    print(f"Confusion matrix saved to {confusion_matrix_path}")
    # plot_training_history(history, MODEL_DIRECTORY)