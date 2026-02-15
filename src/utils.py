"""
Utility functions for camera recognition system.
Helper functions for camera initialization, display, and common operations.
"""

import cv2
import numpy as np


def initialize_camera(camera_id=0, width=640, height=480):
    """
    Initialize camera with specified dimensions.
    
    Args:
        camera_id: Camera device ID (default: 0)
        width: Frame width
        height: Frame height
        
    Returns:
        VideoCapture object
    """
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    print(f"Camera initialized: {width}x{height}")
    return cap


def display_frame(frame, window_name='Frame'):
    """
    Display a frame in a window.
    
    Args:
        frame: Image frame to display
        window_name: Name of the display window
    """
    cv2.imshow(window_name, frame)


def add_text_overlay(frame, text, position=(10, 30), 
                     font=cv2.FONT_HERSHEY_SIMPLEX,
                     font_scale=0.7, color=(0, 255, 0), thickness=2):
    """
    Add text overlay to a frame.
    
    Args:
        frame: Input image frame
        text: Text to display
        position: (x, y) coordinates for text
        font: OpenCV font type
        font_scale: Font size scale
        color: Text color (BGR)
        thickness: Text thickness
        
    Returns:
        Frame with text overlay
    """
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame


def resize_frame(frame, width=None, height=None, interpolation=cv2.INTER_LINEAR):
    """
    Resize frame while maintaining aspect ratio if only one dimension specified.
    
    Args:
        frame: Input image frame
        width: Target width (optional)
        height: Target height (optional)
        interpolation: Interpolation method
        
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width is None:
        aspect_ratio = height / h
        width = int(w * aspect_ratio)
    elif height is None:
        aspect_ratio = width / w
        height = int(h * aspect_ratio)
    
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def draw_fps(frame, fps, position=(10, 30)):
    """
    Draw FPS counter on frame.
    
    Args:
        frame: Input image frame
        fps: Frames per second value
        position: (x, y) coordinates for FPS text
        
    Returns:
        Frame with FPS overlay
    """
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    return frame


def save_frame(frame, filename):
    """
    Save frame to file.
    
    Args:
        frame: Image frame to save
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(filename, frame)
        print(f"Frame saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving frame: {e}")
        return False
