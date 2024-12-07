import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize and load the face detection model
def initialize_face_detection():
    print("Initializing face detection...")
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize and load the recognition model
def initialize_face_recognition():
    print("Loading face recognition model...")
    model = load_model('models/face_recognition_model.h5')
    label_encoder = np.load('models/label_encoder.npy', allow_pickle=True)
    return model, label_encoder

# Function for preprocessing the face region
def preprocess_face(face_roi):
    face_resized = cv2.resize(face_roi, (64, 64))
    face_array = img_to_array(face_resized) / 255.0
    return np.expand_dims(face_array, axis=0)

# Function for recognizing faces in a frame
def process_frame(frame, face_cascade, model, label_encoder):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region of interest (ROI)
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess and predict
        face_array = preprocess_face(face_roi)
        prediction = model.predict(face_array)
        predicted_class = np.argmax(prediction)
        label = label_encoder[predicted_class]

        # Display the label above the rectangle
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Recognize faces from webcam
def recognize_face_from_webcam(face_cascade, model, label_encoder):
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame = process_frame(frame, face_cascade, model, label_encoder)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Exit on pressing ESC
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print("Escape hit, closing...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Recognize faces from video
def recognize_face_from_video(video_path, face_cascade, model, label_encoder):
    print(f"Starting video recognition for: {video_path}")
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame = process_frame(frame, face_cascade, model, label_encoder)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Exit on pressing ESC
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print("Escape hit, closing...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Initialize systems
    face_cascade = initialize_face_detection()
    model, label_encoder = initialize_face_recognition()

    print("Choose the recognition method:")
    print("1. Face recognition via webcam")
    print("2. Face recognition from a video")
    
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        recognize_face_from_webcam(face_cascade, model, label_encoder)
    elif choice == '2':
        video_path = input("Enter the path to the video: ")
        recognize_face_from_video(video_path, face_cascade, model, label_encoder)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
