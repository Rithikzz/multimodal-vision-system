🧠 Multimodal Real-Time Vision System

A real-time camera-based AI system that integrates hand gesture recognition, object detection, and emotion recognition using pre-trained deep learning models.
The system is gesture-controlled, enabling or disabling vision modules dynamically for an interactive user experience.

🚀 Features
✋ Hand Gesture Recognition

Detects the following gestures in real time:

Thumbs Up 👍

Open Palm ✋

Fist ✊

Peace ✌️

Pointing ☝️

Used as a control signal for activating other modules.

📦 Object Detection

Detects common indoor objects using a YOLO-based model:

Person

Cell Phone

Bottle

Laptop

Chair

Book

Cup

Keyboard

Mouse

Bounding boxes with confidence scores are displayed on screen.

🙂 Emotion Recognition

Detects facial emotions (single face per frame):

Happy

Neutral

Sad

Angry

Surprised

Uses a pre-trained CNN for real-time inference.

🧩 Gesture-Controlled Logic

Thumbs Up → Enable object & emotion detection

Fist → Disable all detections

Open Palm → Pause (status only)

This makes the system interactive and intelligent, not just reactive.

🛠️ Tech Stack

Python

OpenCV – camera handling & visualization

MediaPipe – hand gesture recognition

YOLOv8 (Ultralytics) – object detection

TensorFlow / Keras – emotion recognition

NumPy

All models used are pre-trained (no dataset collection or training).

📁 Project Structure
camera-recognition/
│
├── main.py
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── gesture.py
│   ├── object.py
│   ├── emotion.py
│   └── utils.py
│
├── models/
│   ├── yolo.pt
│   └── emotion_model.h5
│
└── assets/
    └── labels.txt


⚠️ Model files are excluded from GitHub using .gitignore.

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/multimodal-vision-system.git
cd multimodal-vision-system

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Model Files

Place the following files manually:

models/yolo.pt

models/emotion_model.h5

▶️ Run the Project
python main.py


Press q to exit.

🎯 Output Example
Gesture: Thumbs Up (0.92)
MODE: ACTIVE

📦 Object: Person (0.95)
📦 Object: Laptop (0.88)

🙂 Emotion: Happy (0.87)
FPS: 18

🧠 Why This Project Is Advanced

Multimodal AI integration

Real-time inference (FPS ≥ 15)

Gesture-controlled system logic

Confidence-based filtering

Clean, explainable architecture

Industry-style use of pre-trained models

🎓 Academic / Interview Explanation

“This project demonstrates a real-time multimodal vision system where hand gestures dynamically control object detection and emotion recognition using pre-trained deep learning models. The focus is on system integration, performance, and reliable inference rather than model training.”

📌 Future Improvements (Optional)

FPS optimization using threading

Temporal smoothing for predictions

Deployment as a desktop or web app

📜 License

This project is for educational and academic use.
