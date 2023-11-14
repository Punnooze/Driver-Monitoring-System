import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid #unique identifier
import os
import time
import tkinter as tk
import pygame
import dlib
from imutils import face_utils

pygame.mixer.init()
audio_file='alert.mp3'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)
img = os.path.join('data', 'images', 'awake.6c29e3cc-6ccf-11ee-baa6-50c2e86bea56.jpg')

last=0
awake=0
drowsy=0
phone=0
sideways=0
count=1
root = tk.Tk()
root.withdraw()


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

ddrowsy=0
dactive=0
dsleep=0
dcount=0
color=(0,0,0)

def compute(ptA, ptB):
    dist=np.linalg.norm(ptA-ptB)
    return dist
def blinked(a,b,c,d,e,f):
    up= compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    if ratio>0.25:
        return 2
    elif ratio>0.21 and ratio <=0.25:
        return 1
    else:
        return 0
w=False
f=False


status=""
while cap.isOpened():
    ret, frame= cap.read()
    results = model(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    w=True
    for face in faces:
        f=True
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0),2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if(left_blink==0 or right_blink==0):
            dsleep+=1
            awake=0
            print("ds",dsleep)
            if(dsleep>6):
                status = "SLEEPING"
                color = (255,0,0)
        elif(left_blink==1 or right_blink==1):
            dsleep=0
            awake=0
            ddrowsy+=1
            print("dd",ddrowsy)
            if(ddrowsy>6):
                status = "DROWSY"
                color = (0,0,255)
        else:
            ddrowsy=0 
            dsleep=0
            awake+=1
            print('da',dactive)
            if(awake>=1):
                status="ACTIVE"
                color=(0,255,0)
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        for n in range(0,68):
            (x,y) = landmarks[n]
            cv2.circle(frame,(x,y),1,(255,255,255),-1)
        if(ddrowsy>=10 or dsleep >=10):
            print("HHH")
            ddrowsy=0
            dsleep=0
            awake=0
            status=""
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            if(awake >= 1):
                pygame.mixer.music.stop()
                print('Stopped')

    if(w and f and status != ""):
        cv2.imshow("YOLO", frame)
        key = cv2.waitKey(1)
        if key==27:
            break
    
    if len(results.xywh[0]) > 0:
        x=0.6
                    # if results.xywh[0][0][5]==17:
            #     x=0.4
        dconf = results.xywh[0][0][4]
        if dconf.item() >= x:
            dclass = results.xywh[0][0][5]
            if dclass == 15:
                if last==0:
                    last==15
                awake+=1
                    # print('a',awake)
                if last!=15:
                    last=15
                    awake=1
                phone=0
                drowsy=0
                sideways=0
            if dclass == 16:
                if last==0:
                    last==16
                if last == 16:
                    drowsy+=1
                        # print('d',drowsy)
                if last != 16:
                    last=16
                    drowsy=0
            if dclass == 17:
                if last==0:
                    last==17
                if last == 17:
                    phone+=1
                    # print('p',phone)
                if last != 17:
                    last=17
                    phone=0
            if dclass == 18:
                if last==0:
                    last==18
                if last == 18:
                    sideways+=1
                        # print('s',sideways)
                if last != 18:
                    last=18
                    sideways=1
            z=int(100/count)
            if phone>=z or sideways>=z or drowsy>=z:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                if count <6:
                    count+=1
                drowsy=0
                phone=0
                sideways=0
                awake=0
            if awake >=5:
                pygame.mixer.music.stop()
                count=1

            cv2.imshow('YOLO', np.squeeze(results.render()))
        else:
            cv2.imshow('YOLO', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break;
    w=False
    f=False
cap.release()
cv2.destroyAllWindows()