import cv2
import time
from logger import save_log
from alert import alert
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import load_model

#screenshot function 
def save_screenshot(frame):

    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")


    filename = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S.jpg"
    )


    path = "screenshots/" + filename


    cv2.imwrite(
        path,
        frame
    )

# Load trained model
model = load_model("models/mask_detector.h5")
last_alert_time = 0
last_screenshot_time = 0


# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Open webcam
cap = cv2.VideoCapture(0)

total_people = 0
mask_count = 0
no_mask_count = 0
frame_count = 0

while True:
    frame_count += 1
    total_people = 0
    mask_count = 0
    no_mask_count = 0

    ret, frame = cap.read()

    if not ret:
        break


    # Convert grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )


    for (x,y,w,h) in faces:

        total_people += 1

        face = frame[y:y+h, x:x+w]


        # Resize same as training
        face = cv2.resize(face, (224,224))

        face = face / 255.0

        face = face.reshape(1,224,224,3)


        prediction = model.predict(face)

        confidence = prediction[0][0]

        if confidence < 0.5:

            label = "Mask"
            color = (0,255,0)
            mask_count += 1


        else:

            label = "No Mask"
            color = (0,0,255)

            no_mask_count += 1


            current_time = time.time()


            if current_time - last_screenshot_time > 10:

                save_screenshot(frame)

                last_screenshot_time = current_time


            current_time = time.time()


            if current_time - last_alert_time > 5:

                alert()

                last_alert_time = current_time

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            color,
            2
        )


        text = f"{label} {confidence*100:.2f}%"

        cv2.putText(
            frame,
            text,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.putText(
    frame,
    f"Total: {total_people}",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255,255,255),
    2
    )

    cv2.putText(
        frame,
        f"Mask: {mask_count}",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"No Mask: {no_mask_count}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )
    if frame_count % 100 == 0:

        save_log(
            total_people,
            mask_count,
            no_mask_count
        )

    cv2.imshow(
        "Face Mask Detection",
        frame
    )


    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()