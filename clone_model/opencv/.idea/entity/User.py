class User:
    def __init__(self, user_id, username, fingerprint, face_image):
        self.user_id = user_id
        self.user_name = username
        self.fingerPrint = fingerprint
        self.face = face_image

    def __str__(self):
        return f"User ID: {self.user_id}, Username: {self.user_name}, FingerPrint: {self.fingerPrint}"
