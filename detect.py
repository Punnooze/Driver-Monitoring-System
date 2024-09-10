import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid #unique identifierpi
import os
import time
import tkinter as tk
import pygame

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
while cap.isOpened():
    ret, frame= cap.read()
    results = model(frame)
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
                print('a',awake)
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
                    print('d',drowsy)
                if last != 16:
                    last=16
                    drowsy=0
            if dclass == 17:
                if last==0:
                    last==17
                if last == 17:
                    phone+=1
                    print('p',phone)
                if last != 17:
                    last=17
                    phone=0
            if dclass == 18:
                if last==0:
                    last==18
                if last == 18:
                    sideways+=1
                    print('s',sideways)
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
cap.release()
cv2.destroyAllWindows()