class FingerPrint:
    def __init__(self, image, features=None):
        self.image = image
        self.features = features

    def extract_features(self, model):
        """
        Trích xuất đặc trưng từ vân tay bằng model dự đoán.
        """
        self.features = model.predict(self.image)
        return self.features

    def __str__(self):
        return f"Hình ảnh vân tay {self.image}, đặc điểm vân tay {self.features})"
