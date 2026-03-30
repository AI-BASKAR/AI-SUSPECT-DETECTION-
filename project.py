import cv2
import os
import face_recognition
import requests
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# ---------------- TELEGRAM ----------------
BOT_TOKEN = "8742074728:AAFKe_XgxTdWOMkgmSUGGtVTpJFm0hLPwd4"
CHAT_ID = "5852552649"

def send_telegram_alert(image_path, message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    with open(image_path, "rb") as img:
        requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": message},
            files={"photo": img}
        )

# ---------------- LOCATION ----------------
def get_location():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5)
        data = r.json()

        loc = data.get("loc","")
        city = data.get("city","")
        region = data.get("region","")
        country = data.get("country","")

        if loc != "":
            lat,lon = loc.split(",")
            maps = f"https://www.google.com/maps?q={lat},{lon}"
        else:
            maps = "Unknown"

        location = f"{city}, {region}, {country}"

        return location, maps

    except:
        return "Unknown","Unknown"

# ---------------- YOLO ----------------
print("Loading YOLO...")
model = YOLO("yolov8n.pt")

# ---------------- LOAD SUSPECTS ----------------
known_encodings = []
known_names = []

dataset_path = "suspects"

for person in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, person)

    if not os.path.isdir(folder):
        continue

    for img in os.listdir(folder):

        path = os.path.join(folder,img)

        image = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(image)

        if len(enc) > 0:
            known_encodings.append(enc[0])
            known_names.append(person)

print("Loaded suspects:",known_names)

# ---------------- DASHBOARD ----------------
root = tk.Tk()
root.title("AI Suspect Tracker")
root.geometry("700x600")
root.configure(bg="#1e1e1e")

title = tk.Label(root,
                 text="AI Suspect Tracker",
                 fg="#00ff88",
                 bg="#1e1e1e",
                 font=("Arial",18,"bold"))
title.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

status = tk.Label(root,
                  text="Camera stopped",
                  fg="white",
                  bg="#1e1e1e",
                  font=("Arial",12))
status.pack(pady=10)

cap = None
running = False
last_alert = 0

# ---------------- CAMERA LOOP ----------------
def update_frame():

    global last_alert

    if not running:
        return

    ret, frame = cap.read()

    if ret:

        results = model(frame)

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])

                if cls == 0:

                    x1,y1,x2,y2 = map(int,box.xyxy[0])

                    person = frame[y1:y2,x1:x2]

                    if person.size == 0:
                        continue

                    rgb = cv2.cvtColor(person,cv2.COLOR_BGR2RGB)

                    faces = face_recognition.face_encodings(rgb)

                    for face in faces:

                        name = "Unknown"

                        if len(known_encodings)>0:

                            dist = face_recognition.face_distance(
                                known_encodings, face)

                            best = np.argmin(dist)

                            if dist[best] < 0.5:
                                name = known_names[best]

                        if name!="Unknown":

                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                            cv2.putText(frame,name,(x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,(0,0,255),2)

                            now = datetime.now().timestamp()

                            if now-last_alert > 10:

                                path="suspect.jpg"
                                cv2.imwrite(path,frame)

                                loc,map_link=get_location()

                                msg=f"🚨 Suspect Detected: {name}\n📍 {loc}\n🗺 {map_link}"

                                send_telegram_alert(path,msg)

                                last_alert=now

                        else:

                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                            cv2.putText(frame,"Unknown",(x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,(0,255,0),2)

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(10,update_frame)

# ---------------- BUTTONS ----------------
def start_camera():
    global cap,running
    cap=cv2.VideoCapture(0)
    running=True
    status.config(text="Camera running")
    update_frame()

def stop_camera():
    global running
    running=False
    if cap:
        cap.release()
    status.config(text="Camera stopped")

start_btn = tk.Button(root,text="START",
                      bg="green",fg="white",
                      command=start_camera)

start_btn.pack(pady=5)

stop_btn = tk.Button(root,text="STOP",
                     bg="red",fg="white",
                     command=stop_camera)

stop_btn.pack(pady=5)

root.mainloop()