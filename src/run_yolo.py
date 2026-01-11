# src/run_yolo.py

import numpy as np
from typing import List, Dict
from ultralytics import YOLO

# Global model instance (loaded once)
_model = None


def get_model():
    """Load YOLOv8n model once (cached globally)."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def run_yolo(frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
    """
    Run YOLO inference on a frame.
    
    Args:
        frame: numpy array of shape (H, W, 3) in RGB format
        conf_threshold: minimum confidence score (0.0 to 1.0). Default 0.25.
        
    Returns:
        List of dicts, each with keys:
            - 'cls': int (class ID: 0=person, 2=car)
            - 'xyxy': tuple of (x1, y1, x2, y2) coordinates
            - 'conf': float (confidence score 0.0 to 1.0)
    """
    model = get_model()
    results = model(frame, verbose=False, conf=conf_threshold)
    
    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = tuple(box.xyxy[0].cpu().numpy().astype(int))
            boxes.append({
                'cls': cls,
                'xyxy': xyxy,
                'conf': conf,
            })
    
    return boxes


def filter_peds(boxes: List[Dict]) -> List[Dict]:
    """
    Filter boxes to only pedestrians (COCO class 0).
    
    Args:
        boxes: List of box dicts from run_yolo()
        
    Returns:
        Filtered list containing only pedestrian boxes
    """
    return [box for box in boxes if box['cls'] == 0]


def filter_cars(boxes: List[Dict]) -> List[Dict]:
    """
    Filter boxes to only cars (COCO class 2).
    
    Args:
        boxes: List of box dicts from run_yolo()
        
    Returns:
        Filtered list containing only car boxes
    """
    return [box for box in boxes if box['cls'] == 2]
