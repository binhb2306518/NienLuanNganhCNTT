"""
src/__init__.py - Khoi tao package ma nguon he thong phat hien vi pham giao thong
================================================================================
Kien truc module huong doi tuong (OOP) cho de tai:
"Tu dong phat hien phuong tien vi pham den tin hieu giao thong"

Sinh vien: Nguyen Thanh Binh - B2306518 - CT466E - DH Can Tho
"""

from .video_loader import VideoLoader
from .detector import Detector, DetectionResult
from .tracker import Tracker, TrackedObject, BaseTracker
from .traffic_light import TrafficLightClassifier, LightState, TrafficLightResult
from .violation_logic import ViolationDetector, ViolationEvent, segments_intersect
from .evidence import EvidenceExtractor

__all__ = [
    # Video handling
    "VideoLoader",
    # Object detection
    "Detector",
    "DetectionResult",
    # Tracking
    "Tracker",
    "TrackedObject",
    "BaseTracker",
    # Traffic light
    "TrafficLightClassifier",
    "LightState",
    "TrafficLightResult",
    # Violation reasoning
    "ViolationDetector",
    "ViolationEvent",
    "segments_intersect",
    # Evidence packaging
    "EvidenceExtractor",
]

__version__ = "1.0.0"

