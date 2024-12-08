class FingerPrint:
    def __init__(self, image, features=None):
        self.image = image  # Hình ảnh vân tay (binary hoặc đường dẫn)
        self.features = features  # Đặc trưng đã trích xuất

    def extract_features(self, model):
        """
        Trích xuất đặc trưng từ vân tay bằng model dự đoán.
        """
        self.features = model.predict(self.image)  # Giả định model trả về vector đặc trưng
        return self.features

    def __str__(self):
        return f"Fingerprint image: {self.image}, Features: {self.features}"
