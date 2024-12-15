from tkinter import messagebox, Tk
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
import time

# Initialize tkinter (to show messagebox)
root = Tk()
root.withdraw()  # Hide main tkinter window

# Load the trained model
model = load_model("clone_model\\opencv\\.idea\models\\signLang_model\\best_model_vgg16.keras")

# Capture the camera feed
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
words = ''  # Save predicted characters

# Set a delay (in seconds) between predictions
prediction_delay = 1.0  # 1-second delay
last_prediction_time = time.time()

print("Press 'q' to quit.")

# Main loop to test the system
while True:
    success, img = cap.read()

    if not success:
        continue

    # Detect hands in the image
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgHeight, imgWidth, _ = img.shape

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            # Resize the cropped hand image while maintaining the aspect ratio
            # Resize the cropped hand image while maintaining the aspect ratio
            aspectRatio = h / w

            if aspectRatio > 1:
                k = 150 / h  # Resize to 150x150
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 150))
                wGap = math.ceil((150 - wCal) / 2)
                imgWhite = np.ones((150, 150, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = 150 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (150, hCal))
                hGap = math.ceil((150 - hCal) / 2)
                imgWhite = np.ones((150, 150, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Normalize the image to [0, 1] before prediction
            imgWhite_norm = imgWhite.astype(np.float32) / 255.0
            imgWhite_norm = np.expand_dims(imgWhite_norm, axis=0)

            # Only predict if the delay has passed
            current_time = time.time()
            if current_time - last_prediction_time > prediction_delay:
                try:
                    # Predict with the model
                    prediction = model.predict(imgWhite_norm)
                    predicted_class = np.argmax(prediction[0])

                    char = 'S' if predicted_class == 1 else 'O'
                    words += char

                    print(f"Predicted Character: {char}")

                    # Check if we detect 'SOS'
                    if len(words) >= 3 and words[-3:] == 'SOS':
                        messagebox.showwarning("SOS ALERT", "SOS! NEED HELP!!")
                        words = ''
                        break
                        

                except Exception as e:
                    print(f"Prediction Error: {e}")

                last_prediction_time = current_time  # Update the last prediction time

        # Show the video stream
        cv2.imshow('Camera Feed', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Clean up after testing
cap.release()
cv2.destroyAllWindows()
