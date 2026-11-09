"""
src/violation_logic.py - Logic phat hien vi pham khong-gian thoi gian
======================================================================
Phat hien phuong tien vuot den do dua tren:
  1. Trang thai den: RED (on dinh >= T_buffer giay)
  2. Vector di chuyen cat vach dung (Stop Line)
  3. Xac nhan boi vach thu 2 (Violation Confirmation Line)

Input:  TrackedObject + TrafficLightResult + cau hinh vach
Output: ViolationEvent hoac None
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import uuid
import time
import numpy as np

from traffic_light import LightState, TrafficLightResult
from tracker import TrackedObject


@dataclass
class ViolationEvent:
    """
    Su kien vi pham duoc xac nhan.

    Attributes:
        violation_id  : ID duy nhat cua su kien (UUID)
        track_id      : ID phuong tien vi pham
        vehicle_class : Loai phuong tien (car, motorcycle, ...)
        timestamp     : Thoi diem xac nhan vi pham (Unix time)
        frame_index   : So frame tai thoi diem vi pham
        status        : "CONFIRMED" sau khi hoi du dieu kien
        evidence_path : Duong dan thu muc bang chung (duoc gan sau)
    """
    violation_id:  str
    track_id:      int
    vehicle_class: str
    timestamp:     float
    frame_index:   int
    status:        str = "CONFIRMED"
    evidence_path: Optional[str] = None


def _cross_product_2d(o, a, b) -> float:
    """
    Tinh tich co (cross product) 2D cua 2 vector OA va OB.
    Dung de xac dinh phia cua diem so voi duong thang.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, p3, p4) -> bool:
    """
    Kiem tra 2 doan thang P1P2 va P3P4 co cat nhau khong.
    Dung de kiem tra xe co vuot qua vach dung khong.

    Args:
        p1, p2: Diem dau/cuoi cua doan 1 (vach dung).
        p3, p4: Diem dau/cuoi cua doan 2 (vector di chuyen xe).
    Returns:
        True neu 2 doan cat nhau.
    """
    d1 = _cross_product_2d(p3, p4, p1)
    d2 = _cross_product_2d(p3, p4, p2)
    d3 = _cross_product_2d(p1, p2, p3)
    d4 = _cross_product_2d(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


class ViolationDetector:
    """
    Kiem tra dieu kien vi pham vuot den do cho tung phuong tien.

    Args:
        stop_line     : Vach dung [[x1,y1],[x2,y2]].
        confirm_line  : Vach xac nhan vi pham (cach vach dung ~50px).
        min_track_len : So diem lich su toi thieu truoc khi danh gia.
        red_stable_frames: So frame den do phai on dinh truoc khi bat vi pham.
    """

    def __init__(
        self,
        stop_line:          list[list[int]],
        confirm_line:       list[list[int]],
        min_track_len:      int   = 10,
        red_stable_frames:  int   = 15,
    ):
        self.stop_line         = stop_line
        self.confirm_line      = confirm_line
        self.min_track_len     = min_track_len
        self.red_stable_frames = red_stable_frames

        # Trang thai noi bo
        self._red_frame_count:  int   = 0  # So frame den do lien tiep
        self._crossed_stop:     set[int] = set()   # Track IDs da cat vach 1
        self._violated:         set[int] = set()   # Track IDs da bi ghi vi pham

    def update_light(self, light: TrafficLightResult):
        """Cap nhat dem frame trang thai den do."""
        if light.state == LightState.RED:
            self._red_frame_count += 1
        else:
            self._red_frame_count = 0

    @property
    def is_red_stable(self) -> bool:
        """Den do da on dinh du so frame chua?"""
        return self._red_frame_count >= self.red_stable_frames

    def check(
        self,
        track:       TrackedObject,
        frame_index: int,
    ) -> Optional[ViolationEvent]:
        """
        Kiem tra mot phuong tien co vi pham khong.

        Args:
            track      : Thong tin phuong tien dang theo doi.
            frame_index: So frame hien tai.
        Returns:
            ViolationEvent neu vi pham duoc xac nhan, None neu chua.
        """
        # Dieu kien 1: Den phai dang do on dinh
        if not self.is_red_stable:
            return None

        # Dieu kien 2: Quy dao du dai de phan tich
        if len(track.history) < self.min_track_len:
            return None

        # Dieu kien 3: Phuong tien chua bi ghi vi pham truoc do
        if track.track_id in self._violated:
            return None

        # Dieu kien 4: Kiem tra xe cat vach dung (vach 1)
        history = list(track.history)
        sl_p1 = tuple(self.stop_line[0])
        sl_p2 = tuple(self.stop_line[1])

        for i in range(len(history) - 1):
            prev_pt = history[i]
            curr_pt = history[i + 1]
            if segments_intersect(sl_p1, sl_p2, prev_pt, curr_pt):
                self._crossed_stop.add(track.track_id)
                break

        # Dieu kien 5: Kiem tra xe vuot vach xac nhan (vach 2)
        if track.track_id in self._crossed_stop:
            cl_p1 = tuple(self.confirm_line[0])
            cl_p2 = tuple(self.confirm_line[1])

            for i in range(len(history) - 1):
                prev_pt = history[i]
                curr_pt = history[i + 1]
                if segments_intersect(cl_p1, cl_p2, prev_pt, curr_pt):
                    self._violated.add(track.track_id)
                    return ViolationEvent(
                        violation_id  = str(uuid.uuid4())[:8],
                        track_id      = track.track_id,
                        vehicle_class = track.class_name,
                        timestamp     = time.time(),
                        frame_index   = frame_index,
                    )

        return None

