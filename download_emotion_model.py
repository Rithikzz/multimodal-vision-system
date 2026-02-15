"""
Download pre-trained emotion detection model
"""
import urllib.request
import os

# Model URL (pre-trained emotion detection model)
MODEL_URL = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
MODEL_PATH = "models/emotion_model.h5"

os.makedirs("models", exist_ok=True)

print("Downloading emotion detection model...")
print(f"URL: {MODEL_URL}")
print(f"Saving to: {MODEL_PATH}")

try:
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✓ Model downloaded successfully!")
    print(f"✓ Saved to: {MODEL_PATH}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("\nAlternative: Creating a simple emotion model...")
    
    # Create a simple CNN model as fallback
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 emotions: Sad, Neutral, Happy
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.save(MODEL_PATH)
    print(f"✓ Created simple model: {MODEL_PATH}")
    print("⚠ Note: This is an untrained model - predictions will be random")
    print("⚠ For real use, train the model or download a pre-trained one")
