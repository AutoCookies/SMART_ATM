import FingerPrint
class User (FingerPrint):
    def __init__(self, user_id, username, fingerprintImage, face_image):
        super().__init__(fingerprintImage)
        self.user_id = user_id
        self.user_name = username
        self.face = face_image

    def __str__(self):
        return f"User ID: {self.user_id}, Username: {self.user_name}, FingerPrint: {self.fingerPrint}"
