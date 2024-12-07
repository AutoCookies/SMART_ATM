import cv2
from pymongo import MongoClient

def collect_data(request):
    # Kết nối đến MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["face_db"]
    collection = db["face_images"]

    # Nhập nhãn cho hình ảnh
    label = request.form.get('label')  # Lấy nhãn từ form nhập liệu trên trang web

    # Mở camera và thu thập dữ liệu
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Press Space to capture")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow("Press Space to capture", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"face_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            with open(img_name, "rb") as image_file:
                # Lưu hình ảnh cùng với nhãn vào MongoDB
                collection.insert_one({"image": image_file.read(), "label": label})
            print(f"{img_name} saved with label: {label}")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
