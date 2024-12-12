import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time
import pygame
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from playsound import playsound
import sys
import speech_recognition as sr
import librosa
import sounddevice as sd
import threading
import tensorflow as tf
load_dotenv()
model_subject = load_model('models\\fingerprint_regconition\\USER.keras')
model_finger = load_model('models\\fingerprint_regconition\\FINGER.keras')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model_face_recognition = load_model('models/face_regconition/my_model2.h5')
label_encoder = np.load('models/face_regconition/label_encoder.npy', allow_pickle=True)

model_voice = load_model("models\\voice_model\\voice_attention_test_reg3.h5")
voice_labels = "models\\voice_model\\label_map.txt"
GREETING_AUDIO_PATH = "openSound.wav"
GOOGBYE_AUDIO_PATH = "goodBye.wav"

img_size = 96
N_MFCC = 13
MAX_LEN = 30
SAMPLE_RATE = 16000
AUDIO_DURATION = 3

# khởi động phát audio
pygame.mixer.init()

label_map = {}
with open(os.path.join("models\\voice_model", "label_map.txt"), "r") as f:
    for line in f:
        label, idx = line.strip().split("\t")
        label_map[int(idx)] = label

def preprocess_image(img_path):
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_array, (img_size, img_size))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def record_audio(duration=AUDIO_DURATION, sample_rate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete")
    return audio.flatten()

def extract_mfcc_from_audio(audio, sample_rate=SAMPLE_RATE):
    audio = librosa.util.normalize(audio)  # Normalize audio
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc.T

def predict_audio_class(audio):
    mfcc = extract_mfcc_from_audio(audio)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model_voice.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return label_map[predicted_class]

def predict_fingerprint(img_path):
    img_array = preprocess_image(img_path)

    subject_prediction = model_subject.predict(img_array)
    subject_id = np.argmax(subject_prediction)

    finger_prediction = model_finger.predict(img_array)
    finger_num = np.argmax(finger_prediction)

    return subject_id, finger_num

def preprocess_face(face_roi):
    face_resized = cv2.resize(face_roi, (64, 64))
    face_array = img_to_array(face_resized) / 255.0
    return np.expand_dims(face_array, axis=0)

def process_frame(frame, face_cascade, model, label_encoder):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]
        face_array = preprocess_face(face_roi)
        prediction = model.predict(face_array)
        predicted_class = np.argmax(prediction)
        label = label_encoder[predicted_class]

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return label
    return None

def recognize_face_from_webcam_with_delay(face_cascade, model, label_encoder, subject_id, cap, delay=7, timeout=20):
    start_time = time.time()

    # Giai đoạn đợi (delay)
    while time.time() - start_time < delay:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame during delay.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Keep your face steady in front of the camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face Recognition - Initializing", frame)
        if cv2.waitKey(1) % 256 == 27:
            return None

    # Giai đoạn nhận diện
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        recognized_label = process_frame(frame, face_cascade, model, label_encoder)
        if recognized_label and str(recognized_label) == str(subject_id):
            cv2.imshow("Face Recognition", frame)
            return recognized_label
        cv2.imshow("Face Recognition", frame)
        if time.time() - start_time > timeout:
            break
        if cv2.waitKey(1) % 256 == 27:
            break

    return None

def start_face_recognition(subject_id, root, cap):
    attempts = 0
    while attempts < 3:
        face_recognized = recognize_face_from_webcam_with_delay(
            face_cascade, model_face_recognition, label_encoder, subject_id, cap, delay=7, timeout=20
        )
        
        if face_recognized and str(face_recognized) == str(subject_id):
            user_info = is_valid_user(subject_id)
            if user_info:
                username = user_info.get("username", "Unknown User")  # Lấy tên người dùng hoặc gán giá trị mặc định
                messagebox.showinfo("Success", f"WELCOME {username}! All checks passed.")
                root.destroy()
                
                show_welcome_window(user_info)  # Truyền username vào
                return True
            else:
                messagebox.showerror("Error", "User not found in database. Access denied.")
                return False
        else:
            attempts += 1
            messagebox.showwarning("Cảnh báo!", f"Lần thử thứ {attempts}/3 thất bại. Xin hãy thử lại.")
    capture_unauthorized_face()
    messagebox.showerror("Lỗi!", "Vượt quá số lần thử cho phép. Truy cập bị từ chối.")
    return False


def capture_unauthorized_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("unauthorized_person\\unauthorized_access.jpg", frame)
        print("Unauthorized access attempt recorded.")
    cap.release()
    cv2.destroyAllWindows()

def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            index, label = line.strip().split()
            label_map[int(index)] = label
    return label_map

def predict_command(command):
    features = np.random.rand(1, 100)

    prediction = model_voice.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return predicted_class

def show_account_balance_window(user_info):
    balance_window = tk.Toplevel()
    balance_window.title("Số Dư Tài Khoản")
    balance_window.geometry("1480x1200")
    balance_window.configure(bg="#E7F9E5")

    # Hiển thị nhãn tiêu đề
    balance_label = tk.Label(balance_window, text="Số dư tài khoản của bạn là:", font=("Arial", 14), bg="#E7F9E5")
    balance_label.pack(pady=20)

    # Lấy số dư từ `user_info`
    balance = user_info.get("balance", 0)  # Mặc định là 0 nếu không tìm thấy

    # Hiển thị giá trị số dư
    balance_value_label = tk.Label(balance_window, text=f"{balance:,} VND", font=("Arial", 16, "bold"), bg="#E7F9E5")
    balance_value_label.pack(pady=10)

    # Nút đóng cửa sổ
    close_button = tk.Button(balance_window, text="Đóng", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=20, command=balance_window.destroy)
    close_button.pack(pady=20)

def show_recharge_window():
    recharge_window = tk.Toplevel()
    recharge_window.title("Nạp Tiền")
    recharge_window.geometry("1480x1200")

    label = tk.Label(recharge_window, text="Chọn phương thức nạp tiền:", bg="#95D5B2", fg="#1B4332", font=("Arial", 14))
    label.pack(pady=20)

    credit_button = tk.Button(recharge_window, text="Nạp qua thẻ tín dụng", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20)
    credit_button.pack(pady=10)
    
    wallet_button = tk.Button(recharge_window, text="Nạp qua ví điện tử", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20)
    wallet_button.pack(pady=10)
    
    transfer_button = tk.Button(recharge_window, text="Nạp qua chuyển khoản ngân hàng", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20)
    transfer_button.pack(pady=10)
    
    back_button = tk.Button(recharge_window, text="Quay lại", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=20, command=recharge_window.destroy)
    back_button.pack(pady=10)

def show_withdrawal_window(user_info):
    withdrawal_window = tk.Toplevel()
    withdrawal_window.title("Rút Tiền Mặt")
    withdrawal_window.geometry("1480x1200")
    withdrawal_window.configure(bg="#E7F9E5")
    
    balance = user_info.get("balance", 0)

    label = tk.Label(withdrawal_window, text="Nhập số tiền muốn rút:", font=("Arial", 14))
    label.pack(pady=20)

    amount_entry = tk.Entry(withdrawal_window, font=("Arial", 14), width=20)
    amount_entry.pack(pady=10)

    balance_label = tk.Label(withdrawal_window, text=f"Số dư hiện tại: {balance}vnđ", font=("Arial", 14), bg="#E7F9E5")
    balance_label.pack(pady=20)

    def withdrawal():
        try:
            withdrawal_amount = int(amount_entry.get().replace("vnđ", "").strip())
        except ValueError:
            messagebox.showerror("Lỗi", "Số tiền nhập vào không hợp lệ. Vui lòng thử lại.")
            return
        
        nonlocal balance  # Đảm bảo có thể sửa biến cục bộ bên ngoài
        if withdrawal_amount > balance:
            messagebox.showerror("Lỗi", "Không thể rút, số dư không đủ.")
        else:
            new_balance = balance - withdrawal_amount
            user_id = user_info.get("user_id")  # Giả định user_info chứa user_id

            # Cập nhật cơ sở dữ liệu
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                raise ValueError("MongoDB URI not found in .env file.")

            client = MongoClient(mongo_uri, server_api=ServerApi('1'))
            db = client['Accounts']
            accounts_collection = db['accounts']

            # Thực hiện cập nhật số dư
            result = accounts_collection.update_one(
                {"user_id": user_id},
                {"$set": {"balance": new_balance}}
            )

            if result.modified_count > 0:
                balance = new_balance  # Cập nhật biến cục bộ
                balance_label.config(text=f"Số dư hiện tại: {balance}vnđ")  # Cập nhật giao diện
                messagebox.showinfo("Thành công", f"Bạn đã rút {withdrawal_amount}vnđ thành công. Số dư mới: {balance}vnđ.")
            else:
                messagebox.showerror("Lỗi", "Không thể cập nhật số dư. Vui lòng thử lại sau.")

    confirm_button = tk.Button(withdrawal_window, text="Xác nhận", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20, command=withdrawal)
    confirm_button.pack(pady=10)

    cancel_button = tk.Button(withdrawal_window, text="Thoát", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20, command=withdrawal_window.destroy)
    cancel_button.pack(pady=10)

    def listen_for_voice_commands():
        audio = record_audio(duration=3)  # Ví dụ: Ghi âm 3 giây
        predicted_label = predict_audio_class(audio)
        print(f"Predicted label: {predicted_label}")

        voice_to_amount = {
            "100k": "100000vnđ",
            "200k": "200000vnđ",
            "500k": "500000vnđ"
        }

        if predicted_label in voice_to_amount:
            amount_entry.delete(0, tk.END)
            amount_entry.insert(0, voice_to_amount[predicted_label])
        elif predicted_label == "thoat":
            withdrawal_window.destroy()
        elif predicted_label == "xoa":
            amount_entry.delete(0, tk.END)
        else:
            print("Lệnh không nhận diện được.")

    listen_button = tk.Button(withdrawal_window, text="Lắng nghe lệnh", font=("Arial", 14), width=20, command=listen_for_voice_commands)
    listen_button.pack(pady=20)

# Show the payment window (Thanh Toán Trực Tuyến)
def show_payment_window(user_info):
    def sort_bills(criteria):
        if criteria == "time":
            user_info['bills'].sort(key=lambda bill: bill['start_date'])
        elif criteria == "amount":
            user_info['bills'].sort(key=lambda bill: bill['total_amount'])
        refresh_bills()

    def refresh_bills():
        for widget in bills_frame.winfo_children():
            widget.destroy()

        for bill in user_info['bills']:
            bill_text = f"{bill["bill_type"]} | ID: {bill["bill_id"]} | " \
                        f"Start: {bill["start_date"]} | Deadline: {bill["deadline"]} | Amount: {bill["total_amount"]}"
            bill_label = tk.Label(bills_frame, text=bill_text, font=("Arial", 12), bg="#F0FFF0")
            bill_label.pack(fill=tk.X, padx=10, pady=5)

            
    def pay_credit():
        total_amount = sum(bill['total_amount'] for bill in user_info['bills'] if not bill['is_paid'])
        balance = user_info.balance

        if balance >= total_amount:
            user_info.balance -= total_amount

            for bill in user_info.get_unpaid_bills():
                bill.is_paid = True
                bill.paid_amount = bill.total_amount

            messagebox.showinfo("Thanh toán thành công", f"Đã thanh toán tổng số tiền {total_amount}.\nSố dư còn lại: {user_info.balance}")

            refresh_bills()

        else:
            messagebox.showerror("Thanh toán thất bại", f"Số dư không đủ. Cần {total_amount}, nhưng chỉ còn {balance}")

    payment_window = tk.Toplevel()
    payment_window.title("Thanh Toán Trực Tuyến")
    payment_window.geometry("1480x1200")
    payment_window.configure(bg="#E7F9E5")

    label = tk.Label(payment_window, text="Chọn phương thức thanh toán:", font=("Arial", 14))
    label.pack(pady=20)

    # Buttons for payment methods
    credit_button = tk.Button(payment_window, text="Thanh toán qua thẻ tín dụng", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20, command=pay_credit)
    credit_button.pack(pady=10)

    wallet_button = tk.Button(payment_window, text="Thanh toán qua ví điện tử", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20)
    wallet_button.pack(pady=10)

    transfer_button = tk.Button(payment_window, text="Thanh toán qua chuyển khoản ngân hàng", bg="#95D5B2", fg="#1B4332", font=("Arial", 14), width=20)
    transfer_button.pack(pady=10)

    back_button = tk.Button(payment_window, text="Quay lại", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=20, command=payment_window.destroy)
    back_button.pack(pady=10)

    # Frame to display bills
    bills_label = tk.Label(payment_window, text="Danh sách hóa đơn:", font=("Arial", 16), bg="#E7F9E5")
    bills_label.pack(pady=10)

    sort_frame = tk.Frame(payment_window)
    sort_frame.pack(fill=tk.X)

    tk.Label(sort_frame, text="Sắp xếp theo:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    btn_sort_time = tk.Button(sort_frame, text="Thời gian", font=("Arial", 12), command=lambda: sort_bills("time"))
    btn_sort_time.pack(side=tk.LEFT, padx=5)
    btn_sort_amount = tk.Button(sort_frame, text="Số tiền", font=("Arial", 12), command=lambda: sort_bills("amount"))
    btn_sort_amount.pack(side=tk.LEFT, padx=5)

    bills_frame = tk.Frame(payment_window)
    bills_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    refresh_bills()

def show_welcome_window(user_info):
    welcome_window = tk.Toplevel()
    welcome_window.title("Welcome")
    welcome_window.geometry("1480x1200")
    welcome_window.configure(bg="#E7F9E5")
    

    
    canvas_frame = tk.Frame(welcome_window, bg="#E7F9E5")
    canvas_frame.pack(expand=True, pady=20)

    canvas = tk.Canvas(canvas_frame, width=600, height=100, bg="#E7F9E5", highlightthickness=0)
    canvas.pack()

    canvas.create_rectangle(
        15, 15, 585, 95,
        fill="#A6D7A3", outline=""
    )

    canvas.create_rectangle(
        10, 10, 580, 90,
        fill="#E7F9E5", outline="#2D6A4F", width=3
    )
    
    username = user_info.get("username", "Unknown User")
    
    message_label = tk.Label(
        canvas_frame,
        text=f"Chào {username}, vui lòng chọn giao dịch!",
        font=("Arial", 20, "bold"),
        fg="#2D6A4F",
        bg="#E7F9E5"
    )
    
    canvas.create_window(300, 50, window=message_label)

    button_frame = tk.Frame(welcome_window, bg="#E7F9E5")
    button_frame.pack(expand=True, padx=20, pady=20)

    left_frame = tk.Frame(button_frame, bg="#E7F9E5")
    left_frame.pack(side=tk.LEFT, padx=50)

    withdraw_button = tk.Button(left_frame, text="Rút tiền mặt", font=("Arial", 14), width=35, bg="#95D5B2", fg="#1B4332", command=lambda:show_withdrawal_window(user_info))
    withdraw_button.pack(pady=10)
    check_info_button = tk.Button(left_frame, text="Tra cứu thông tin", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=35)
    check_info_button.pack(pady=10)
    deposit_button = tk.Button(left_frame, text="Nộp tiền mặt", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=35)
    deposit_button.pack(pady=10)
    balance_button = tk.Button(left_frame, text="Số dư tài khoản", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=35, command=lambda:show_account_balance_window(user_info))
    balance_button.pack(pady = 10)

    right_frame = tk.Frame(button_frame, bg="#E7F9E5")
    right_frame.pack(side=tk.RIGHT, padx=50)

    transfer_button = tk.Button(right_frame, text="Chuyển tiền", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=35)
    transfer_button.pack(pady=10)
    recharge_button = tk.Button(right_frame, text="Nạp tiền ĐTDĐ", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=35, command=show_recharge_window)
    recharge_button.pack(pady=10)
    bill_payment_button = tk.Button(right_frame, text="Thanh toán hóa đơn", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=35, command=lambda:show_payment_window(user_info))
    bill_payment_button.pack(pady=10)
    exit_button = tk.Button(right_frame, text="Thoát", font=("Arial", 14), bg="#95D5B2", fg="#1B4332", width=35, command=sys.exit)
    exit_button.pack(pady=10)

    def listen_for_voice_commands():
        audio = record_audio(duration=AUDIO_DURATION)
        predicted_label = predict_audio_class(audio)
        print(f"Predicted label: {predicted_label}")
        
        if str(predicted_label) == "rut_tien":
            print("CHÀO MỪNG BẠN ĐẾN VỚI GIAO DIỆN RÚT TIỀN")
            show_withdrawal_window(user_info)
        elif str(predicted_label) == "so_du":
            print("CHÀO MỪNG BẠN ĐẾN VỚI GIAO DIỆN SỐ DƯ")
            show_account_balance_window(user_info)
        elif str(predicted_label) == 'nap_tien':
            print("CHÀO MỪNG BẠN ĐẾN VỚI GIAO DIỆN NẠP TIỀN")
            show_recharge_window()
        elif str(predicted_label) == 'thanh_toan':
            print("CHÀO MỪNG BẠN ĐẾN VỚI GIAO DIỆN THANH TOÁN TRỰC TUYẾN")
            show_payment_window()
        elif str(predicted_label) == 'thoat':
            welcome_window.destroy()
        else:
            print('CÁC LỆNH KHÁC')

    listen_for_voice_commands_button = tk.Button(welcome_window, text="Start Listening for Commands", font=("Arial", 14), command=listen_for_voice_commands)
    listen_for_voice_commands_button.pack(pady=20)

def is_valid_user(subject_id):
    """
    Kiểm tra xem subject_id có tồn tại trong cơ sở dữ liệu hay không.
    Trả về thông tin người dùng nếu tồn tại, ngược lại trả về None.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MongoDB URI not found in .env file.")

    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    db = client['Accounts']
    accounts_collection = db['accounts']

    # Lấy thông tin người dùng từ cơ sở dữ liệu
    user = accounts_collection.find_one({"user_id": int(subject_id)})
    return user  # Trả về thông tin người dùng hoặc None nếu không tồn tại


def fingerprint_upload_window(cap):
    finger_print_window = tk.Toplevel()
    finger_print_window.title("Fingerprint Upload")
    finger_print_window.geometry("400x500")

    label = tk.Label(finger_print_window, text="Xin hãy xác nhận vân tay của bạn", font=("Arial", 16))
    label.pack(pady=20)

    image_label = tk.Label(finger_print_window)
    image_label.pack(pady=10)

    scanning_label = tk.Label(finger_print_window, text="", font=("Arial", 14)) 
    scanning_label.pack()

    prediction_label = tk.Label(finger_print_window, text="", font=("Arial", 14), fg="green")
    prediction_label.pack(pady=10)

    def upload_fingerprint():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo
            
            scanning_label.config(text="Đang quét vân tay, xin vui lòng đợi trong giây lát...", fg="orange")
            root.after(2000, lambda: process_fingerprint(file_path, cap))

    def process_fingerprint(file_path, cap):
        subject_id, finger_num = predict_fingerprint(file_path)
        if is_valid_user(subject_id):
            messagebox.showinfo("Fingerprint Prediction", f"Subject ID: {subject_id}, Finger Type: {finger_num}")
            prediction_label.config(text=f"Subject ID: {subject_id}, Finger Type: {finger_num}")
            if start_face_recognition(subject_id, root, cap):
                root.destroy()
                show_welcome_window(str(subject_id))
        else:
            messagebox.showerror("Error", "Subject ID not found in database.")


    upload_button = tk.Button(finger_print_window, text="Upload Fingerprint", font=("Arial", 14), command=upload_fingerprint)
    upload_button.pack(pady=10)

    exit_button = tk.Button(finger_print_window, text="Exit", font=("Arial", 14), command=root.quit)
    exit_button.pack(pady=10)
    
def play_goodbye():
    try:
        pygame.mixer.music.load(GOOGBYE_AUDIO_PATH)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"Error playing goodbye audio: {e}")

def play_greeting():
    try:
        pygame.mixer.music.load(GREETING_AUDIO_PATH)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing welcome audio: {e}")

def detect_face_and_greet():
    cap = cv2.VideoCapture(0)

    detected = False
    start_time = None
    no_face_start_time = None
    face_present = False
    bat_dau_detected = False  # Biến cờ để theo dõi lệnh đã được phát hiện

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face_present = True
            no_face_start_time = None  # Reset no face timer

            if not detected:
                detected = True
                start_time = time.time()

            if time.time() - start_time >= 2 and not bat_dau_detected:
                play_greeting()
                audio = record_audio(duration=AUDIO_DURATION)
                predicted_label = predict_audio_class(audio)
                print(f"Predicted label: {predicted_label}")

                if predicted_label == "bat_dau":
                    fingerprint_upload_window(cap)
                    bat_dau_detected = True

                time.sleep(3)  # Optional delay

                detected = False

        else:
            if face_present:
                if no_face_start_time is None:
                    no_face_start_time = time.time()

                if time.time() - no_face_start_time >= 5:
                    play_goodbye()
                    face_present = False
                    # break

            detected = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def begin_GUI(root):
    root.title("Face Detection Greeting")
    tk.Label(root, text="CHÀO MỪNG BẠN ĐÃ ĐẾN VỚI ATM THÔNG MINH!").pack(pady=20)
    threading.Thread(target=detect_face_and_greet, daemon=True).start()
    
root = tk.Tk()
root.title("SMART ATM SYSTEM")
root.geometry("500x400")

begin_GUI(root)

root.mainloop()