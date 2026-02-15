import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load emotion model once (compile=False to avoid optimizer issues)
emotion_model = load_model("models/emotion_model.h5", compile=False)

# Model outputs 7 classes, we map to 3
EMOTIONS_7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTIONS = ["Sad", "Neutral", "Happy"]

# Mapping from 7 classes to 3 classes
EMOTION_MAP = {
    "angry": "Sad",
    "disgust": "Sad",
    "fear": "Sad",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Happy",
    "neutral": "Neutral"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    emotion = "None"
    confidence = 0.0
    bbox = None

    # Only process the largest face (most likely to be the actual person)
    if len(faces) > 0:
        # Get the largest face by area
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Filter out faces that are too small or oddly shaped
        aspect_ratio = w / float(h)
        if 0.7 < aspect_ratio < 1.3 and w > 50:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))

            preds = emotion_model.predict(face, verbose=0)
            
            # Map 7-class predictions to 3 classes
            mapped_preds = {"Sad": 0.0, "Neutral": 0.0, "Happy": 0.0}
            for i, prob in enumerate(preds[0]):
                emotion_7 = EMOTIONS_7[i]
                emotion_3 = EMOTION_MAP[emotion_7]
                mapped_preds[emotion_3] += prob
            
            # Get emotion with highest probability
            emotion = max(mapped_preds, key=mapped_preds.get)
            confidence = float(mapped_preds[emotion])
            
            # Only return detection if confidence is reasonable
            if confidence > 0.3:
                bbox = (x, y, x+w, y+h)
            else:
                emotion = "None"
                confidence = 0.0

    return emotion, confidence, bbox
