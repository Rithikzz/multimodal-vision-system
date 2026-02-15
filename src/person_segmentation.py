import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download model if not exists
MODEL_PATH = 'models/selfie_segmenter.tflite'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'

if not os.path.exists(MODEL_PATH):
    print("Downloading person segmentation model...")
    os.makedirs('models', exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Person segmentation model downloaded!")

# Create image segmenter
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    output_category_mask=True
)

segmenter = vision.ImageSegmenter.create_from_options(options)


def extract_person_contour(frame, bbox):
    """
    Extract accurate person silhouette using MediaPipe segmentation.
    
    Args:
        frame: Input BGR image
        bbox: Tuple of (x1, y1, x2, y2) bounding box from YOLO
        
    Returns:
        contour: Largest external contour of the person
        mask: Binary mask of the person
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    # Ensure bbox is within frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Extract ROI and convert to RGB for MediaPipe
    person_roi = frame[y1:y2, x1:x2]
    if person_roi.size == 0:
        return None, None
    
    roi_h, roi_w = person_roi.shape[:2]
    rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
    
    # Perform segmentation
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask
    
    # Convert mask to numpy array (0 = background, 1 = person)
    mask_data = category_mask.numpy_view()
    person_mask = (mask_data > 0).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur to smooth edges
    person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)
    
    # Threshold to get binary mask
    _, person_mask = cv2.threshold(person_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    if not contours:
        return None, None
    
    # Get the largest contour (main person body)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out very small contours (noise)
    area = cv2.contourArea(largest_contour)
    if area < 100:  # Minimum area threshold
        return None, None
    
    # Smooth the contour using approximation
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    smooth_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    return smooth_contour, person_mask


def draw_person_silhouette(frame, bbox, color=(0, 255, 0), thickness=3):
    """
    Draw person silhouette on frame using accurate segmentation.
    
    Args:
        frame: Input BGR image (will be modified)
        bbox: Tuple of (x1, y1, x2, y2) bounding box
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        success: Boolean indicating if silhouette was drawn
    """
    x1, y1, x2, y2 = bbox
    
    # Extract person contour
    contour, mask = extract_person_contour(frame, bbox)
    
    if contour is None:
        # Fallback: draw simple bounding box if segmentation fails
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return False
    
    # Offset contour to match frame coordinates
    contour_offset = contour + [x1, y1]
    
    # Draw the silhouette
    cv2.drawContours(frame, [contour_offset], -1, color, thickness)
    
    return True
