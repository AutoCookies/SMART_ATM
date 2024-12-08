from pymongo import MongoClient
from gridfs import GridFS
import base64
import os

def encode_file_to_base64(file_path):
    """
    Convert a file (image, video) to Base64 encoding.
    """
    with open(file_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return encoded

def save_file_to_gridfs(fs, file_path, file_name):
    """
    Save a file to GridFS.
    """
    with open(file_path, "rb") as file:
        file_id = fs.put(file, filename=file_name)
    return file_id

try:
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client['Accounts']
    accounts_collection = db['accounts']
    fs = GridFS(db)

    data_dir = 'data\\users_data'

    print("Connected to the database")

    for idx, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)

        if os.path.isdir(folder_path):

            # Encode fingerprint images to Base64
            finger_images = [
                encode_file_to_base64(os.path.join(folder_path, f))
                for f in os.listdir(folder_path) if f.endswith('.BMP')
            ]

            # Save videos to GridFS
            video_files = [
                save_file_to_gridfs(fs, os.path.join(folder_path, video), f"{video}")
                for video in os.listdir(folder_path) if video.endswith('.mp4')
            ]

            # Encode face image to Base64
            face_files = [
                encode_file_to_base64(os.path.join(folder_path, f))
                for f in os.listdir(folder_path) if f.endswith('.jpg')
            ]

            # Construct account data dictionary
            account_data = {
                "user_id": idx,
                "username": f"user_{idx}",
                "fingerprints": [{"image": img, "features": None} for img in finger_images],
                "videos": [str(video) for video in video_files],  # Store GridFS file IDs
                "face_image": face_files[0] if face_files else None,
                "balance": 1000.0,
                "debt": 200.0
            }

            # Insert account data into MongoDB
            accounts_collection.insert_one(account_data)

            print(f"Account {idx} data successfully inserted into MongoDB.")

except Exception as e:
    print(f"Failed to save data into MongoDB: {str(e)}")