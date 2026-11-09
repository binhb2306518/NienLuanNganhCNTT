"""
src/traffic_light.py - Nhan dien trang thai den tin hieu giao thong
====================================================================
Ho tro 2 phuong phap:
  1. HSV color analysis (nhanh, khong can GPU)
  2. YOLO classification (chinh xac hon, can GPU)

Input:  Frame numpy (H, W, 3) BGR
Output: TrafficLightState dict
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import cv2
import numpy as np


class LightState(str, Enum):
    RED     = "RED"
    YELLOW  = "YELLOW"
    GREEN   = "GREEN"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrafficLightResult:
    """
    Ket qua phan tich trang thai den tin hieu.

    Attributes:
        state      : Trang thai hien tai (RED/YELLOW/GREEN/UNKNOWN)
        confidence : Do tin cay uoc tinh [0.0, 1.0]
        box        : Bounding box den [x1, y1, x2, y2] (None neu dung ROI co dinh)
        method     : Phuong phap su dung ("hsv" hoac "yolo_cls")
    """
    state:      LightState
    confidence: float
    box:        Optional[list[int]] = None
    method:     str = "hsv"


class TrafficLightClassifier:
    """
    Phan loai trang thai den tin hieu bang phan tich mau HSV.

    Args:
        roi (tuple): Vung ROI co dinh (x1, y1, x2, y2) tinh theo pixel.
                     Neu None, phan tich toan bo frame.
        min_ratio  : Ti le pixel mau toi thieu de xac dinh trang thai.
    """

    # Nguong mau HSV (H: 0-179, S: 0-255, V: 0-255) trong OpenCV
    RED_RANGES = [
        (np.array([0,  120, 70]),  np.array([10,  255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ]
    YELLOW_RANGE = (np.array([15,  120, 70]), np.array([35,  255, 255]))
    GREEN_RANGE  = (np.array([45,  100, 70]), np.array([85,  255, 255]))

    def __init__(self, roi: Optional[tuple[int, int, int, int]] = None, min_ratio: float = 0.12):
        self.roi       = roi         # (x1, y1, x2, y2) pixel
        self.min_ratio = min_ratio

    def classify(self, frame: np.ndarray) -> TrafficLightResult:
        """
        Phan loai trang thai den tu frame.

        Args:
            frame: Anh BGR numpy (H, W, 3).
        Returns:
            TrafficLightResult voi trang thai va do tin cay.
        """
        # Cat vung ROI neu co cau hinh
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_img = frame[y1:y2, x1:x2]
        else:
            roi_img = frame

        if roi_img.size == 0:
            return TrafficLightResult(LightState.UNKNOWN, 0.0)

        # Chuyen sang khong gian mau HSV
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # Tinh ti le pixel cho tung mau
        red_mask = (
            cv2.inRange(hsv, self.RED_RANGES[0][0], self.RED_RANGES[0][1]) |
            cv2.inRange(hsv, self.RED_RANGES[1][0], self.RED_RANGES[1][1])
        )
        yellow_mask = cv2.inRange(hsv, self.YELLOW_RANGE[0], self.YELLOW_RANGE[1])
        green_mask  = cv2.inRange(hsv, self.GREEN_RANGE[0],  self.GREEN_RANGE[1])

        red_ratio    = cv2.countNonZero(red_mask)    / total_pixels
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
        green_ratio  = cv2.countNonZero(green_mask)  / total_pixels

        # Xac dinh trang thai
        ratios = {
            LightState.RED:    red_ratio,
            LightState.YELLOW: yellow_ratio,
            LightState.GREEN:  green_ratio,
        }
        best_state, best_ratio = max(ratios.items(), key=lambda x: x[1])

        if best_ratio < self.min_ratio:
            return TrafficLightResult(LightState.UNKNOWN, 0.0)

        return TrafficLightResult(
            state      = best_state,
            confidence = min(best_ratio / 0.4, 1.0),  # Chuan hoa [0,1]
            box        = list(self.roi) if self.roi else None,
            method     = "hsv",
        )

