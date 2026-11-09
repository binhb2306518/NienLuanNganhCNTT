"""
src/detector.py - Wrapper YOLO cho phat hien phuong tien va den tin hieu
=========================================================================
Input:  Frame numpy (H, W, 3) BGR
Output: List[DetectionResult] - cac doi tuong duoc phat hien
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """
    Ket qua phat hien mot doi tuong.

    Attributes:
        class_id   : Index lop (0=car, 1=motorcycle, 2=bus, 3=truck, 4=traffic_light)
        class_name : Ten lop
        confidence : Do tin cay [0.0, 1.0]
        box        : Bounding box [x1, y1, x2, y2] tinh theo pixel
        center     : Tam bounding box (cx, cy)
        bottom_center: Diem giua canh duoi (cx, y2) - Vi tri uoc tinh banh xe
    """
    class_id:      int
    class_name:    str
    confidence:    float
    box:           list[int]        # [x1, y1, x2, y2]
    center:        tuple[int, int] = field(init=False)
    bottom_center: tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        self.center        = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.bottom_center = ((x1 + x2) // 2, y2)


class Detector:
    """
    Wrapper cho mo hinh YOLO (Ultralytics).
    Ho tro ca YOLOv8 va YOLOv11 voi FP16 (AMP).

    Args:
        model_path   (str)  : Duong dan file .pt cua mo hinh.
        conf_thresh  (float): Nguong do tin cay toi thieu.
        iou_thresh   (float): Nguong IoU cho NMS.
        device       (str)  : Thiet bi chay: "cuda:0" hoac "cpu".
        use_fp16     (bool) : Bat Mixed Precision de tiet kiem VRAM.
        classes      (list) : Loc chi phat hien cac lop nay (None = tat ca).
    """

    CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "traffic_light"]

    def __init__(
        self,
        model_path:  str,
        conf_thresh: float = 0.45,
        iou_thresh:  float = 0.45,
        device:      str   = "cuda:0",
        use_fp16:    bool  = True,
        classes:     Optional[list[int]] = None,
    ):
        self.model_path  = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.device      = device
        self.use_fp16    = use_fp16
        self.classes     = classes
        self._model      = None

    def load(self):
        """
        Tai model vao bo nho GPU.
        Goi ham nay truoc khi bat dau pipeline.
        """
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        self._model.to(self.device)

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        """
        Chay inference YOLO tren mot frame.

        Args:
            frame: Anh BGR numpy (H, W, 3).
        Returns:
            List cac DetectionResult.
        """
        if self._model is None:
            raise RuntimeError("Chua goi Detector.load() - hay tai mo hinh truoc.")

        results = self._model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            half=self.use_fp16,
            classes=self.classes,
            verbose=False,
        )

        detections: list[DetectionResult] = []
        if not results:
            return detections

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(DetectionResult(
                    class_id   = cls_id,
                    class_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else str(cls_id),
                    confidence = float(box.conf[0]),
                    box        = [int(v) for v in box.xyxy[0].tolist()],
                ))

        return detections

