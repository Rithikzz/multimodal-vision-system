import cv2
import mediapipe as mp
import math
import numpy as np
import urllib.request
import os

# Initialize MediaPipe Hands using the tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download model if not exists
MODEL_PATH = 'models/hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    os.makedirs('models', exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(options)


def get_distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on frame manually."""
    h, w, c = frame.shape
    
    # Define connections (hand skeleton)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmarks
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


def detect_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = detector.detect(mp_image)

    gesture = "None"
    confidence = 0.0

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        index_tip = hand_landmarks[8]
        index_mcp = hand_landmarks[5]
        middle_tip = hand_landmarks[12]
        middle_mcp = hand_landmarks[9]
        ring_tip = hand_landmarks[16]
        ring_mcp = hand_landmarks[13]
        pinky_tip = hand_landmarks[20]
        pinky_mcp = hand_landmarks[17]
        wrist = hand_landmarks[0]

        # Check if fingers are extended (tip higher than knuckle)
        index_up = index_tip.y < index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        ring_up = ring_tip.y < ring_mcp.y
        pinky_up = pinky_tip.y < pinky_mcp.y
        thumb_up = thumb_tip.y < thumb_ip.y

        # Count extended fingers
        fingers_extended = sum([index_up, middle_up, ring_up, pinky_up])

        # Thumbs Up: Only thumb extended, other fingers closed
        if thumb_up and fingers_extended == 0:
            gesture = "Thumbs Up"
            confidence = 0.92

        # Open Palm: All fingers extended
        elif fingers_extended >= 4 and thumb_up:
            gesture = "Open Palm"
            confidence = 0.95

        # Fist: All fingers closed
        elif fingers_extended == 0 and not thumb_up:
            gesture = "Fist"
            confidence = 0.90

        # Draw landmarks
        draw_hand_landmarks(frame, hand_landmarks)

    return gesture, confidence
