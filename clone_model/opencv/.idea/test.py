import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Đang lắng nghe...")

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source, timeout=5)

        # Google API nhận diện giọng nói
        text = recognizer.recognize_google(audio_data, language="vi-VN")
        print(f"Google API nhận diện được: {text}")

        # Chuyển đổi các cụm số thành định dạng chuẩn
        formatted_text = convert_text_to_numbers(text)

        return formatted_text

    except sr.UnknownValueError:
        return "Không thể nhận diện giọng nói."

    except sr.RequestError:
        return "Không thể kết nối Google API."

# Chuyển đổi các cụm từ phức tạp thành số chuẩn
def convert_text_to_numbers(text):
    vietnamese_numbers = {
        "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
        "mười": 10, "trăm": 100, "nghìn": 1000, "triệu": 1000000
    }

    words = text.split(" ")
    result = 0
    temp = 0

    for word in words:
        if word in vietnamese_numbers:
            number = vietnamese_numbers[word]
            
            if number >= 100:
                temp *= number
            else:
                temp += number
            
        elif word == "triệu":
            result += temp * 1000000
            temp = 0
        elif word == "nghìn":
            result += temp * 1000
            temp = 0

    result += temp

    return result

speech_text = recognize_speech()
print(f"Kết quả sau tiền xử lý: {speech_text}")