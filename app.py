# import tkinter as tk
# import customtkinter as ctk
# import torch
# import numpy as np
# import cv2
# from PIL import Image, ImageTk

# app = tk.Tk()
# app.geometry("600x600")
# app.title("DDS")
# ctk.set_appearance_mode("dark")

# vidFrame = tk.Frame(height=480, width=600)
# vidFrame.pack()
# vid = ctk.CTkLabel(vidFrame)
# vid.pack()

# counter = 0
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

# cap = cv2.VideoCapture(0)


# def detect():
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = model(frame)
#     img = np.squeeze(results.render())
    
#     # if len(results.xywh[0]) > 0:
#     #     dconf = results.xywh[0][0][4]
#     #     if dconf.item() >= 0.75:
            
    
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     imgarr = Image.fromarray(img)
#     imgtk = ImageTk.PhotoImage(imgarr)
#     vid.imgtk = imgtk
#     vid.configure(image=imgtk)
#     vid.after(10, detect)


# detect()
# app.mainloop()





import tkinter as tk
import customtkinter as ctk
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk


app = tk.Tk()
app.geometry("600x600")
app.title("DDS")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=480, width=600)
vidFrame.pack()
vid=ctk.CTkLabel(vidFrame)
vid.pack()

counter =0
# def reset_counter():
#     global counter
#     counter=0
# resetButton = ctk.CTkButton(height=40, width=120, text_font=("Arial",20), text="Reset Counter", command=reset_counter, text_color="white", fg_color="teal" )
# resetButton.pack()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

cap = cv2.VideoCapture(0)
def detect():
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img=np.squeeze(results.render())
    f=0
    if len(results.xywh[0]) >0:
        dconf=results.xywh[0][0][4]
        dclass=results.xywh[0][0][5]
        if dconf.item()<0.75:
            f=1

    if f!=1:            
        imgarr = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(imgarr)
        vid.imgtk = imgtk
        vid.configure(image=imgtk)
        vid.after(10,detect)

    else:
        f=0
        vid.after(10,detect)



detect()


app.mainloop()