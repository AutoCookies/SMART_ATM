class User:
    def __init__(self, user_id, username, fingerprints, face_image, face_videos):
        """
        fingerprints: Danh sách các đối tượng FingerPrint
        face_image: Hình ảnh gương mặt (binary hoặc đường dẫn)
        """
        self.user_id = user_id
        self.username = username
        self.fingerprints = fingerprints  # Danh sách đối tượng FingerPrint
        self.face_image = face_image  # Hình ảnh gương mặt
        self.face_videos = face_videos
    def __str__(self):
        fingerprints_str = "\n".join([str(fp) for fp in self.fingerprints])
        return (
            f"User ID: {self.user_id}, Username: {self.username}, "
            f"Face Image: {self.face_image}\nFingerprints:\n{fingerprints_str}"
        )
