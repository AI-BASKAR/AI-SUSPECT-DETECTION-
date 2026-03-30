import cv2
import os

name = "suspect"
folder = f"suspects/{name}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord("s"):
        cv2.imwrite(f"{folder}/{count}.jpg", frame)
        count += 1
        print("Saved image", count)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
