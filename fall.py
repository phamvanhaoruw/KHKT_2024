#Import các thư viện cần thiết
#==============================================#
#-------------thư viện chung--------------------#
import os
import cv2
import time
import threading
#-------------thư viện phất hiện té ngã---------#
import cvzone
import math
from ultralytics import YOLO
#-------------thư viện nhận diện cảm xúc-----------#
from keras_preprocessing.image import img_to_array
from keras.models import load_model
#-------------thư viện nhận diện khuôn mặt---------#
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import sqlite3
#-------------thư viện cho việc phát âm thanh------#
import pygame
#-------------thư viện cho việc gửi thông báo-------#
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#-------------thư viện cho chyển đổi giống nói-------#
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
#-------------thư viện cho phần cứng---------------#
# import serial
# arduino = serial.Serial('COM5', 9600, timeout = 0.01 )#uno
# arduino2 = serial.Serial('COM6', 9600, timeout = 0.01 )#nano
# #==============================================#
def speak(text):
    tts = gTTS(text=text, lang='vi')
    tts.save("speech.mp3")
    playsound("speech.mp3")
    os.remove("speech.mp3")
#==============================================#
def sound(mp):
    pygame.mixer.init()
    pygame.mixer.music.load(mp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
#==============================================#
def email():
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    username = "tuanleanh2882k9@gmail.com"
    password = "wuol dxcm ertn zbiz"

    sender_email = "tuanleanh2882k9@gmail.com"
    receiver_email = "phamvanhao3107@gmail.com"
    subject = "CẢNH BÁO"
    body = "CÓ NGƯỜI BỊ TÉ"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email đã được gửi thành công!")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi gửi email: {e}")
def fall_detect():
    cap = cv2.VideoCapture("fall.mp4")
    model = YOLO('yolov8s.pt')
    classnames = []

    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    fall_start_time = None  # Thời gian bắt đầu phát hiện té ngã
    fall_threshold_time = 5  # Thời gian tối thiểu để xác nhận té ngã
    fall_confirmed = False  # Cờ trạng thái để xác nhận té ngã

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (980, 740))
        results = model(frame)

        fall_detected = False

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                height = y2 - y1
                width = x2 - x1
                threshold = height - width

                if conf > 80 and class_detect == 'person':
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                    if threshold < 0:
                        fall_detected = True
                        if fall_start_time is None:
                            fall_start_time = time.time()  # Bắt đầu đếm thời gian té ngã
                    else:
                        fall_start_time = None  # Reset nếu không còn trạng thái té ngã

        # Reset fall_start_time nếu không phát hiện té ngã trong frame hiện tại
        if not fall_detected:
            fall_start_time = None

        # Xác nhận cú ngã nếu kéo dài lâu hơn ngưỡng thời gian
        if fall_detected and fall_start_time is not None:
            fall_duration = time.time() - fall_start_time
            if fall_duration >= fall_threshold_time:
                fall_confirmed = True
                cvzone.putTextRect(frame, 'Fall Confirmed', [10, 50], thickness=2, scale=2)
                #cvzone.putTextRect(frame, f'Fall Detected: {fall_duration:.2f}s', [10, 50], thickness=2, scale=2)
                speak("Cảnh báo hệ thống phát hiện người bị té")
                email()
                sound("coibaodong.mp3")
                # for i in range(0, 5):  # Gửi email 5 lần
                email()

            else:
                cvzone.putTextRect(frame, f'Fall Detected: {fall_duration:.2f}s', [10, 50], thickness=2, scale=2)
        else:
            fall_confirmed = False  # Reset nếu trạng thái té ngã kết thúc

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()
#==============================================#
fall_detect()