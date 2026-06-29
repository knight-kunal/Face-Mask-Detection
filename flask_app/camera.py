import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# Load Model
# ==========================
model = load_model("../models/mask_detector.h5")

# ==========================
# Face Detector
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

# Memory Variables (For Stability)
last_face = None
missing_frames = 0
prediction_history = []

def generate_frames():
    global last_face, missing_frames, prediction_history
    
    while True:
        success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Haar Cascade ko aur sensitive kiya taaki mask ke sath bhi chehra pakde
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05, 
            minNeighbors=4,
            minSize=(60, 60)
        )

        # ---------------------------------------------------
        # FIX 1: FACE MEMORY (Blinking rokne ke liye)
        # ---------------------------------------------------
        if len(faces) > 0:
            # Agar ek se zyada face hain, toh sabse bada face lo
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            last_face = faces[0]
            missing_frames = 0
            faces_to_process = [last_face]
        else:
            # Agar mask ki wajah se face miss ho jaye, toh 10 frames tak purana box use karo
            if last_face is not None and missing_frames < 10:
                faces_to_process = [last_face]
                missing_frames += 1
            else:
                faces_to_process = []
                prediction_history = []  # Koi face nahi toh history clear kar do

        for (x, y, w, h) in faces_to_process:
            # Padding
            pad = 40
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # BGR to RGB and Preprocessing
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_preprocessed = preprocess_input(face_resized)
            face_expanded = np.expand_dims(face_preprocessed, axis=0)

            # Prediction
            prediction = model.predict(face_expanded, verbose=0)
            confidence = float(prediction[0][0])

            # ---------------------------------------------------
            # FIX 2: PREDICTION SMOOTHING (Fluctuation rokne ke liye)
            # ---------------------------------------------------
            prediction_history.append(confidence)
            if len(prediction_history) > 5:
                prediction_history.pop(0)  # Puraane predictions hatao
            
            avg_confidence = np.mean(prediction_history)

            # Labeling Logic based on Averaged Confidence
            if avg_confidence < 0.5:
                label = "Mask"
                color = (0, 255, 0) # Green
                score = (1 - avg_confidence) * 100
            else:
                label = "No Mask"
                color = (0, 0, 255) # Red
                score = avg_confidence * 100

            # Draw Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Put Text
            cv2.putText(
                frame,
                f"{label} {score:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )