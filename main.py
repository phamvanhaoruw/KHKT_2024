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
import serial
arduino = serial.Serial('COM3', 9600, timeout = 0.01 )#uno
arduino2 = serial.Serial('COM4', 9600, timeout = 0.01 )#nano
#==============================================#
def gard(command):
    arduino.write(f"{command}\n".encode())
    time.sleep(0.5) 
    response = arduino.readline().decode().strip()
    # print(1)
def gard2(command):
    arduino2.write(f"{command}\n".encode())
    time.sleep(0.5) 
    response = arduino2.readline().decode().strip()
    # print(2)

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
#==============================================#
def sound(mp):
    pygame.mixer.init()
    pygame.mixer.music.load(mp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
#==============================================#
def lis():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Nói bắt đầu để mở hệ thống nhận diện khuôn mặt")
        print("Nói bắt đầu để mở hệ thống nhận diện khuôn mặt...")
        recognizer.adjust_for_ambient_noise(source) 
        audio_data = recognizer.listen(source)    

        try:
            text = recognizer.recognize_google(audio_data, language="vi-VN")  # Sử dụng Tiếng Việt
            return text
        except sr.UnknownValueError:
            speak("Tôi không nghe bạn nói gì bạn có thể nói lại được không")
            print("Tôi không nghe bạn nói gì bạn có thể nói lại được không")
        except sr.RequestError:
            print("Có lỗi xảy ra trong quá trình yêu cầu nhận diện.")
#==============================================#
def speak(text):
    tts = gTTS(text=text, lang='vi')
    tts.save("speech.mp3")
    playsound("speech.mp3")
    os.remove("speech.mp3")
#==============================================#
# def fall_detect():
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     model = YOLO('yolov8s.pt')
#     classnames = []

#     with open('classes.txt', 'r') as f:
#         classnames = f.read().splitlines()

#     fall_start_time = None  # Thời gian bắt đầu phát hiện té ngã
#     fall_threshold_time = 5  # Thời gian tối thiểu để xác nhận té ngã
#     fall_confirmed = False  # Cờ trạng thái để xác nhận té ngã

#     while True:
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (980, 740))
#         results = model(frame)

#         fall_detected = False

#         for info in results:
#             parameters = info.boxes
#             for box in parameters:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 confidence = box.conf[0]
#                 class_detect = int(box.cls[0])
#                 class_detect = classnames[class_detect]
#                 conf = math.ceil(confidence * 100)

#                 height = y2 - y1
#                 width = x2 - x1
#                 threshold = height - width

#                 if conf > 80 and class_detect == 'person':
#                     cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
#                     cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

#                     if threshold < 0:
#                         fall_detected = True
#                         if fall_start_time is None:
#                             fall_start_time = time.time()  # Bắt đầu đếm thời gian té ngã
#                     else:
#                         fall_start_time = None  # Reset nếu không còn trạng thái té ngã

#         # Reset fall_start_time nếu không phát hiện té ngã trong frame hiện tại
#         if not fall_detected:
#             fall_start_time = None

#         # Xác nhận cú ngã nếu kéo dài lâu hơn ngưỡng thời gian
#         if fall_detected and fall_start_time is not None:
#             fall_duration = time.time() - fall_start_time
#             if fall_duration >= fall_threshold_time:
#                 fall_confirmed = True
#                 cvzone.putTextRect(frame, 'Fall Confirmed', [10, 50], thickness=2, scale=2)
#                 speak("Cảnh báo phát hệ thống phát hiện người bị té")
#                 sound("coibaodong.mp3")
#                 for i in range(0, 5):  # Gửi email 5 lần
#                     email()

#             else:
#                 cvzone.putTextRect(frame, f'Fall Detected: {fall_duration:.2f}s', [10, 50], thickness=2, scale=2)
#         else:
#             fall_confirmed = False  # Reset nếu trạng thái té ngã kết thúc

#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('t'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
# #==============================================#
def faceRecognition():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read('Model/FaceModel.yml')

    def getProfile(Id):
        conn = sqlite3.connect('Model/FaceDatabase.db')
        query = "SELECT * FROM People WHERE ID=" + str(Id)
        cursor = conn.execute(query)
        profile = None
        for row in cursor:
            profile = row
        conn.close()
        return profile

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    font = cv2.FONT_HERSHEY_COMPLEX

    count_frame = 0  #Bat khung hinh dau tien de mo cua
    door_open = False

    while True:  # hiển thị liên tục
        ret, img = cap.read()  # lấy dữ liệu từ webcam
        # ret trả về true nếu truy cập thành công
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # chuyển ảnh về ảnh sáng
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            nbr_predicted, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf <70:
                profile = getProfile(nbr_predicted) #Lay ten cua nguoi nhan dien duoc
                if profile != None:
                    cv2.putText(img, "" + str(profile[1]), (x + 10, y), font, 1, (0, 255, 0), 1);
                    if ((str(profile[1]) == "Van Hao" or str(profile[1]) == "Khanh Hung") and door_open == False):
                        gard2(1)
                        door_open = True
                        speak("Đã mở cửa")
                        print("Open door!")
            else:
                cv2.putText(img, "Unknown",  (x, y + h + 30), font, 0.4, (0, 255, 0), 1);
        cv2.imshow('Face Detector', img)
        if (cv2.waitKey(1) == ord('q') or door_open == True):
            break
    cap.release()
    cv2.destroyAllWindows()

#==============================================# 
def emotion_Recognition():
    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface.xml')
    classifier = load_model(r'Model\EmotionModel.h5')
    emotion_labels = ['Happy', 'Neutral', 'Surprise']

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    happy_device = False
    suprise_device = False
    countImg = 0
    # start_time = time.time()
    temp = 28
    fan_switch_time = time.time()
    light_switch_time = time.time()
    while True:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            #print(data)
            temp = int(data)
            
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_classifier.detectMultiScale(gray)    
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        label_position = (20, 35)
        cv2.putText(frame, str(temp), label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (x, y, w, h) in faces:  
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                # print(prediction)
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
                if ((happy_device == False) and (label == 'Happy')):  # Bat/tat den
                    countImg += 1
                    light_switch_now = time.time() - light_switch_time
                    if (countImg >= 10 and light_switch_now > 4):
                        device_happy_on()
                        speak("bạn đã bật đèn phòng bằng nhận diện cảm xúc")
                        happy_device = True
                        countImg = 0
                        light_switch_time = time.time()
                elif ((happy_device == True) and (label == 'Happy')):
                    countImg += 1
                    light_switch_now = time.time() - light_switch_time
                    if (countImg >= 10 and light_switch_now > 4):
                        device_happy_off()
                        speak("bạn đã tắt đèn phòng bằng nhận diện cảm xúc")
                        happy_device = False
                        countImg = 0
                        light_switch_time = time.time()

                if ((suprise_device == False) and (label == 'Surprise')):  # Bat/tat quat
                    countImg += 1
                    fan_switch_now = time.time() - fan_switch_time
                    if (countImg >= 10 and fan_switch_now > 4):
                        device_suprise_on()
                        speak("bạn đã bật quạt phòng bằng nhận diện cảm xúc")
                        suprise_device = True
                        countImg = 0    
                        fan_switch_time = time.time()

                elif ((suprise_device == True) and (label == 'Surprise')):
                    countImg += 1
                    fan_switch_now = time.time() - fan_switch_time
                    if (countImg >= 10 and fan_switch_now > 4):
                        device_suprise_off()
                        speak("bạn đã tắt quạt phòng bằng nhận diện cảm xúc")
                        suprise_device = False  
                        countImg = 0
                        fan_switch_time = time.time()
                
                if ((suprise_device == False) and (temp>32)):  # Bat/tat quat tự động
                    # countImg += 1
                    fan_switch_now = time.time() - fan_switch_time
                    if ( fan_switch_now > 4):
                        device_suprise_on()
                        speak("Quạt đã tự động bật do hệ thống thấy nhiệt độ đang tăng cao")
                        suprise_device = True
                        # countImg = 0    
                        fan_switch_time = time.time()

                elif ((suprise_device == True) and (temp<20)):
                    # countImg += 1
                    fan_switch_now = time.time() - fan_switch_time
                    if (fan_switch_now > 4):
                        device_suprise_off()
                        speak("Quạt đã tự động tắt do hệ thống thấy nhiệt độ đang rất thấp")
                        suprise_device = False  
                        countImg = 0
                        fan_switch_time = time.time()
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()   
    cv2.destroyAllWindows()

#=============================== Welcome to HappyHome =======================

print ("Welcome to HappyHome...!")
while True:
    text=lis()
    print("Đang Nhận Diện Giọng Nói")
    if text == "bắt đầu":
        sound("ok.mp3")
        print("Ok")
        speak("Nhận diện khuôn mặt của bạn để vào nhà")
        break
faceRecognition()
gard2(2)  #Dong cua bang servo
speak("Chào mừng bạn đã về nhà")
print ("You are inside House")

def device_happy_on():
    gard(3)
    speak("Bạn đã bật đèn phòng bằng nhận diện cảm xúc")         #Bat/tat den
def device_happy_off():
    gard(4)
    speak("Bạn đã tắt đèn phòng bằng nhận diện cảm xúc")

def device_suprise_on():
    gard(5)
    speak("Bạn đã bật quạt bằng nhận diện cảm xúc")      #Bat bat/tat quat
def device_suprise_off():
    gard(6)
    speak("Bạn tắt quạt bằng nhận diện cảm xúc")

emotion_Recognition()
#==========================================================================
