"""
src/tracker.py - Interface chung cho Object Tracker (ByteTrack / BoT-SORT)
===========================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Hiện thực hóa mẫu thiết kế Chiến lược (Strategy Pattern) cho bài toán
    theo dõi đa mục tiêu (Multi-Object Tracking - MOT). Cung cấp giao diện trừu
    tượng BaseTracker cho phép hoán đổi liền mạch giữa:
      1. ByteTrack (ưu tiên tốc độ, liên kết box điểm thấp D_low)
      2. BoT-SORT (bù chuyển động camera CMC + hiệu chỉnh Kalman w, h)
    mà không phải thay đổi bất kỳ dòng mã logic vi phạm nào ở các tầng sau.

Input:
    - Frame ảnh numpy.ndarray (H, W, 3).
    - List[DetectionResult] hoặc luồng inference tracker từ YOLO.

Output:
    - List[TrackedObject]: Danh sách các đối tượng kèm Track ID bền vững,
      lịch sử quỹ đạo tọa độ tiếp đất (bottom_center) và vector vận tốc.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

try:
    from src.detector import DetectionResult, Detector
except ImportError:
    from detector import DetectionResult, Detector

logger = logging.getLogger("Tracker")


@dataclass
class TrackedObject:
    """
    Thông tin đối tượng đang được theo dõi qua các khung hình liên tiếp.

    Attributes:
        track_id (int): Mã định danh duy nhất do thuật toán theo dõi gán.
        class_id (int): Mã lớp đối tượng (0=car, 1=motorcycle, ...).
        class_name (str): Tên lớp phương tiện.
        confidence (float): Độ tin cậy phát hiện ở frame gần nhất.
        box (List[int]): Tọa độ hộp bao [x1, y1, x2, y2] ở frame hiện tại.
        bottom_center (Tuple[int, int]): Tọa độ đáy hộp bao (cx, y2).
        history (Deque[Tuple[int, int]]): Hàng đợi 2 đầu lưu vết vị trí đáy hộp bao
            qua tối đa N frame gần nhất (mặc định 50 điểm).
        frame_count (int): Số khung hình mà đối tượng đã được theo dõi.
        velocity (Tuple[float, float]): Vector vận tốc di chuyển tức thời (dx, dy).
        last_frame_idx (int): Chỉ số frame cuối cùng đối tượng được quan sát.
    """
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: List[int]
    bottom_center: Tuple[int, int]
    history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=50))
    frame_count: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_frame_idx: int = 0

    def update_position(self, box: List[int], confidence: float, frame_idx: int) -> None:
        """
        Cập nhật vị trí mới, tính toán vận tốc tức thời và bổ sung điểm vào quỹ đạo.

        Args:
            box: Tọa độ [x1, y1, x2, y2] mới.
            confidence: Độ tin cậy detection mới.
            frame_idx: Chỉ số khung hình hiện tại.
        """
        self.box = box
        self.confidence = confidence
        x1, y1, x2, y2 = box
        new_bottom = ((x1 + x2) // 2, y2)

        # Tính toán vector vận tốc (dx, dy)
        dx = float(new_bottom[0] - self.bottom_center[0])
        dy = float(new_bottom[1] - self.bottom_center[1])
        self.velocity = (dx, dy)

        self.bottom_center = new_bottom
        self.history.append(new_bottom)
        self.frame_count += 1
        self.last_frame_idx = frame_idx


class BaseTracker(ABC):
    """
    Giao diện trừu tượng chuẩn (Abstract Base Class) cho mọi thuật toán Tracker.
    """

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: Optional[List[DetectionResult]] = None,
    ) -> List[TrackedObject]:
        """
        Cập nhật trạng thái theo dõi cho khung hình hiện tại.

        Args:
            frame: Ảnh BGR numpy (H, W, 3).
            frame_idx: Chỉ số khung hình (int).
            detections: Danh sách detection (tùy chọn nếu tracker tích hợp với model).

        Returns:
            List[TrackedObject]: Danh sách đối tượng đang được theo dõi có gán ID.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Xóa toàn bộ lịch sử quỹ đạo và reset trạng thái bên trong."""
        pass


class UltralyticsTrackerWrapper(BaseTracker):
    """
    Hiện thực hóa BaseTracker thông qua giao diện `track()` của Ultralytics.
    Hỗ trợ chuyển đổi cấu hình `bytetrack.yaml` hoặc `botsort.yaml`.

    Attributes:
        detector (Detector): Thể hiện lớp Detector đang sở hữu mô hình YOLO.
        algorithm (str): "bytetrack" hoặc "botsort".
        track_buffer (int): Số khung hình tối đa lưu vết đối tượng mất tín hiệu.
    """

    def __init__(
        self,
        detector: Detector,
        algorithm: str = "bytetrack",
        track_buffer: int = 30,
        history_maxlen: int = 50,
    ) -> None:
        self.detector = detector
        self.algorithm = algorithm.lower().strip()
        self.track_buffer = track_buffer
        self.history_maxlen = history_maxlen

        if self.algorithm not in ("bytetrack", "botsort"):
            logger.warning(
                f"Thuật toán '{algorithm}' không phổ biến; mặc định fallback về 'bytetrack'."
            )
            self.algorithm = "bytetrack"

        self.tracker_config = f"{self.algorithm}.yaml"
        self._active_tracks: Dict[int, TrackedObject] = {}

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: Optional[List[DetectionResult]] = None,
    ) -> List[TrackedObject]:
        """
        Thực hiện tracking sử dụng cơ chế model.track() của Ultralytics.
        """
        if not self.detector.is_loaded:
            self.detector.load()

        model = self.detector.raw_model
        results = model.track(
            source=frame,
            conf=self.detector.conf_thresh,
            iou=self.detector.iou_thresh,
            device=self.detector.device,
            half=self.detector.use_fp16,
            classes=self.detector.classes,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        current_frame_tracks: List[TrackedObject] = []
        seen_ids = set()

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.float().cpu().tolist()
            xyxys = boxes.xyxy.int().cpu().tolist()

            for track_id, cls_id, conf, xyxy in zip(ids, clss, confs, xyxys):
                seen_ids.add(track_id)
                if hasattr(model, "names") and cls_id in model.names:
                    cname = model.names[cls_id]
                elif cls_id < len(self.detector.CLASS_NAMES):
                    cname = self.detector.CLASS_NAMES[cls_id]
                else:
                    cname = str(cls_id)

                if track_id in self._active_tracks:
                    obj = self._active_tracks[track_id]
                    obj.update_position(xyxy, conf, frame_idx)
                else:
                    x1, y1, x2, y2 = xyxy
                    bottom = ((x1 + x2) // 2, y2)
                    obj = TrackedObject(
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cname,
                        confidence=conf,
                        box=xyxy,
                        bottom_center=bottom,
                        history=deque(maxlen=self.history_maxlen),
                        frame_count=1,
                        last_frame_idx=frame_idx,
                    )
                    obj.history.append(bottom)
                    self._active_tracks[track_id] = obj

                current_frame_tracks.append(obj)

        # Dọn dẹp các track đã biến mất quá số frame track_buffer
        stale_ids = [
            tid
            for tid, tobj in self._active_tracks.items()
            if (frame_idx - tobj.last_frame_idx) > self.track_buffer
        ]
        for tid in stale_ids:
            del self._active_tracks[tid]

        return current_frame_tracks

    def reset(self) -> None:
        """Xóa danh sách track đang lưu trữ."""
        self._active_tracks.clear()


class Tracker:
    """
    Facade Tracker cung cấp điểm truy cập thống nhất cho toàn bộ hệ thống.
    Cho phép chọn nhanh thuật toán thông qua cấu hình chuỗi (config.yaml).

    Args:
        algorithm (str): "bytetrack" hoặc "botsort".
        detector (Optional[Detector]): Thể hiện detector nếu dùng Ultralytics backend.
        track_buffer (int): Số khung hình lưu vết khi che khuất.
        history_maxlen (int): Độ dài hàng đợi lưu tọa độ quỹ đạo.
    """

    def __init__(
        self,
        algorithm: str = "bytetrack",
        detector: Optional[Detector] = None,
        track_buffer: int = 30,
        history_maxlen: int = 50,
    ) -> None:
        self.algorithm = algorithm.lower().strip()
        self.detector = detector
        self.track_buffer = track_buffer
        self.history_maxlen = history_maxlen

        self._active_tracks: Dict[int, TrackedObject] = {}
        self._strategy: Optional[BaseTracker] = None

        if self.detector is not None:
            self._strategy = UltralyticsTrackerWrapper(
                detector=self.detector,
                algorithm=self.algorithm,
                track_buffer=self.track_buffer,
                history_maxlen=self.history_maxlen,
            )

    def set_detector(self, detector: Detector) -> None:
        """Gán hoặc cập nhật detector cho tracker."""
        self.detector = detector
        self._strategy = UltralyticsTrackerWrapper(
            detector=self.detector,
            algorithm=self.algorithm,
            track_buffer=self.track_buffer,
            history_maxlen=self.history_maxlen,
        )

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        detections: Optional[List[DetectionResult]] = None,
    ) -> List[TrackedObject]:
        """
        Cập nhật tracking cho frame hiện tại.
        """
        if self._strategy is not None:
            tracks = self._strategy.update(frame, frame_idx, detections)
            self._active_tracks = {t.track_id: t for t in tracks}
            return tracks

        # Fallback tracking đơn giản theo centroid distance nếu chưa nạp Ultralytics tracker
        tracks = self._heuristic_tracking(detections or [], frame_idx)
        return tracks

    def _heuristic_tracking(
        self, detections: List[DetectionResult], frame_idx: int
    ) -> List[TrackedObject]:
        """Thuật toán liên kết khoảng cách Euclide phụ trợ khi chạy unit test độc lập."""
        current_tracks: List[TrackedObject] = []
        for det in detections:
            # Tìm track gần nhất
            best_id = None
            best_dist = 60.0  # Ngưỡng pixel liên kết
            for tid, tobj in self._active_tracks.items():
                dist = np.hypot(
                    det.bottom_center[0] - tobj.bottom_center[0],
                    det.bottom_center[1] - tobj.bottom_center[1],
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                obj = self._active_tracks[best_id]
                obj.update_position(det.box, det.confidence, frame_idx)
            else:
                new_id = (max(self._active_tracks.keys()) + 1) if self._active_tracks else 1
                obj = TrackedObject(
                    track_id=new_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    box=det.box,
                    bottom_center=det.bottom_center,
                    history=deque(maxlen=self.history_maxlen),
                    frame_count=1,
                    last_frame_idx=frame_idx,
                )
                obj.history.append(det.bottom_center)
                self._active_tracks[new_id] = obj

            current_tracks.append(obj)
        return current_tracks

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Truy xuất đối tượng theo Track ID."""
        return self._active_tracks.get(track_id)

    @property
    def active_tracks(self) -> List[TrackedObject]:
        """Danh sách đối tượng đang hoạt động."""
        return list(self._active_tracks.values())

    def reset(self) -> None:
        """Reset toàn bộ tracker."""
        self._active_tracks.clear()
        if self._strategy is not None:
            self._strategy.reset()
