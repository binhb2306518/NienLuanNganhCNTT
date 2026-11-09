"""
src/tracker.py - Interface chung cho Object Tracker (ByteTrack / BoT-SORT)
===========================================================================
Input:  List[DetectionResult] + frame
Output: List[TrackedObject] - doi tuong co Track ID va quy dao
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import numpy as np

from detector import DetectionResult


@dataclass
class TrackedObject:
    """
    Doi tuong duoc theo doi qua cac frame.

    Attributes:
        track_id      : ID duy nhat duoc gan boi Tracker
        class_id      : Index lop doi tuong
        class_name    : Ten lop doi tuong
        confidence    : Do tin cay detection moi nhat
        box           : Bounding box [x1, y1, x2, y2]
        bottom_center : Diem giua canh duoi (vi tri banh xe uoc tinh)
        history       : Deque chua lich su vi tri (bottom_center) qua cac frame
        frame_count   : So frame da xuat hien
    """
    track_id:      int
    class_id:      int
    class_name:    str
    confidence:    float
    box:           list[int]
    bottom_center: tuple[int, int]
    history:       deque = field(default_factory=lambda: deque(maxlen=50))
    frame_count:   int = 0

    def update_history(self):
        """Cap nhat lich su vi tri sau moi frame."""
        self.history.append(self.bottom_center)
        self.frame_count += 1


class Tracker:
    """
    Interface chung bao gom ByteTrack va BoT-SORT (qua Ultralytics).
    Co the doi Tracker ma khong thay doi logic vi pham.

    Args:
        algorithm  (str): "bytetrack" hoac "botsort"
        track_buffer (int): So frame giu ID khi mat dau (bi che khuat)
    """

    def __init__(self, algorithm: str = "bytetrack", track_buffer: int = 30):
        self.algorithm    = algorithm.lower()
        self.track_buffer = track_buffer
        self._active_tracks: dict[int, TrackedObject] = {}

    def update(
        self,
        detections: list[DetectionResult],
        frame: np.ndarray,
        detector_model=None,  # Ultralytics YOLO model (dung built-in tracker)
    ) -> list[TrackedObject]:
        """
        Cap nhat trang thai tracker voi cac detection moi nhat.

        Luu y: Tracker se duoc tich hop truc tiep vao Ultralytics YOLO
        bang cach dung model.track() thay vi model.predict(). Ham nay
        la skeleton de dinh nghia interface; implementation se goi
        Ultralytics tracking API.

        Args:
            detections : Danh sach detection trong frame hien tai.
            frame      : Frame hien tai (numpy BGR).
            detector_model: Ultralytics YOLO model (tuy chon).
        Returns:
            List TrackedObject voi Track ID duoc gan.
        """
        # NOTE: Implementation thuc su se su dung:
        # results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        # Skeleton nay dinh nghia interface.
        raise NotImplementedError(
            "Su dung Detector.model.track() de tich hop tracker cua Ultralytics. "
            "Xem vi du trong scripts/run_baseline.py"
        )

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Lay thong tin mot track theo ID."""
        return self._active_tracks.get(track_id)

    @property
    def active_tracks(self) -> list[TrackedObject]:
        """Danh sach cac track dang hoat dong."""
        return list(self._active_tracks.values())

