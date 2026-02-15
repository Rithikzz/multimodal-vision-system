from ultralytics import YOLO

# Load model once
model = YOLO("models/yolo.pt")

# Allowed classes (COCO) - Indoor relevant objects
ALLOWED = {"cell phone", "bottle", "laptop", "chair", "book", "cup", "tv"}


def detect_objects(frame):
    # Lower confidence threshold for more sensitive detection
    results = model(frame, conf=0.2, iou=0.6, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in ALLOWED:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

    return detections
