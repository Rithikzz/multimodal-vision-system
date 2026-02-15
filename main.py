import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

import cv2
import time
import numpy as np
from src.gesture import detect_gesture
from src.object import detect_objects
from src.emotion import detect_emotion
from src.person_segmentation import draw_person_silhouette


def draw_ui_panel(frame, h, w):
    """Draw an interactive UI panel with instructions and status."""
    # Create semi-transparent overlay for header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "MULTIMODAL VISION SYSTEM", (20, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
    
    # Instructions box at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-150), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Instructions
    instructions = [
        "CONTROLS:",
        "👍 Thumbs Up: Emotion Detection",
        "✋ Open Palm: Object Detection", 
        "✊ Fist: DISABLE All",
        "Press 'Q' to Quit"
    ]
    
    y_pos = h - 130
    for i, text in enumerate(instructions):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        font_size = 0.6 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        cv2.putText(frame, text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        y_pos += 25


def draw_status_indicators(frame, gesture, g_conf, object_enabled, emotion_enabled, fps):
    """Draw status indicators showing current system state."""
    h, w = frame.shape[:2]
    
    # Gesture status (top-left)
    gesture_color = (0, 255, 0) if gesture != "None" else (100, 100, 100)
    cv2.putText(frame, f"Gesture: {gesture}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    cv2.putText(frame, f"Confidence: {g_conf:.2%}", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 1)
    
    # Status indicators (top-right)
    status_x = w - 320
    
    # Object Detection Status
    obj_status = "ON" if object_enabled else "OFF"
    obj_color = (0, 255, 0) if object_enabled else (0, 0, 255)
    cv2.rectangle(frame, (status_x, 50), (status_x + 150, 80), obj_color, -1)
    cv2.putText(frame, f"Objects: {obj_status}", (status_x + 10, 72), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Emotion Detection Status
    emo_status = "ON" if emotion_enabled else "OFF"
    emo_color = (0, 255, 0) if emotion_enabled else (0, 0, 255)
    cv2.rectangle(frame, (status_x + 160, 50), (status_x + 310, 80), emo_color, -1)
    cv2.putText(frame, f"Emotion: {emo_status}", (status_x + 170, 72), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # FPS counter (top-right corner)
    fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

object_enabled = False
emotion_enabled = False

# FPS calculation
prev_time = time.time()
fps = 0

print("\n" + "="*60)
print("🎥 MULTIMODAL VISION SYSTEM STARTED")
print("="*60)
print("\n📋 Instructions:")
print("   👍 Thumbs Up  → Enable emotion detection")
print("   ✋ Open Palm  → Enable object detection")
print("   ✊ Fist       → Disable all detection")
print("   Q            → Quit application")
print("\n" + "="*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    h, w = frame.shape[:2]
    
    # Draw UI panels first
    draw_ui_panel(frame, h, w)
    
    gesture, g_conf = detect_gesture(frame)

    # Gesture-based control
    if gesture == "Thumbs Up":
        if not emotion_enabled or object_enabled:  # Only print when state changes
            print("😊 Emotion Detection ENABLED")
        object_enabled = False
        emotion_enabled = True

    elif gesture == "Open Palm":
        if not object_enabled or emotion_enabled:  # Only print when state changes
            print("🎯 Object Detection ENABLED")
        object_enabled = True
        emotion_enabled = False

    elif gesture == "Fist":
        if object_enabled or emotion_enabled:  # Only print when state changes
            print("❌ All Detection DISABLED")
        object_enabled = False
        emotion_enabled = False
    
    # Draw status indicators
    draw_status_indicators(frame, gesture, g_conf, object_enabled, emotion_enabled, fps)

    y_offset = 80

    # Object Detection
    if object_enabled:
        detections = detect_objects(frame)
        obj_count = len(detections)
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            if label == "person":
                # Draw accurate person silhouette using ML-based segmentation
                success = draw_person_silhouette(frame, (x1, y1, x2, y2), color=(0, 255, 0), thickness=3)
                
                # Draw label with background
                label_text = f"{label.upper()} ({conf:.0%})"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # Draw regular box for other objects with thicker lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                label_text = f"{label.upper()} ({conf:.0%})"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Show detection count
        if obj_count > 0:
            cv2.putText(frame, f"Objects Detected: {obj_count}", (w - 300, h - 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Emotion Detection
    if emotion_enabled:
        emotion, e_conf, bbox = detect_emotion(frame)
        if bbox:
            x1, y1, x2, y2 = bbox
            
            # Color-code emotions
            if emotion == "Happy":
                emotion_color = (0, 255, 0)  # Green
            elif emotion == "Sad":
                emotion_color = (0, 0, 255)  # Red
            else:
                emotion_color = (255, 165, 0)  # Orange for Neutral
            
            # Draw thicker face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), emotion_color, 3)
            
            # Draw emotion label with background
            label_text = f"{emotion.upper()} ({e_conf:.0%})"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + text_size[0] + 10, y1), emotion_color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Multimodal Vision System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("\n🛑 Shutting down system...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ System closed successfully\n")