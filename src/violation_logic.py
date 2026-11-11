"""
src/violation_logic.py - Logic phat hien vi pham khong-gian thoi gian
======================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Xác định sự kiện phương tiện vi phạm vượt đèn tín hiệu dựa trên mô hình hình học
    không - thời gian hai vạch ảo (Dual Virtual Lines):
      1. Điều kiện thời gian: Đèn tín hiệu phải ở trạng thái RED ổn định >= N frame.
      2. Điều kiện hình học 1: Quỹ đạo tiếp đất của xe cắt qua Vạch Dừng (Stop Line).
      3. Điều kiện hình học 2: Xe tiếp tục cắt qua Vạch Xác Nhận (Confirmation Line)
         đi sâu vào giao lộ thay vì chỉ dừng sát vạch.
      4. Cơ chế loại trừ: Hỗ trợ kiểm tra hướng rẽ phải nếu làn đường cho phép rẽ.

Input:
    - TrackedObject (chứa lịch sử tọa độ quỹ đạo bánh xe).
    - TrafficLightResult (trạng thái đèn tín hiệu hiện tại).
    - Cấu hình tọa độ 2 vạch ảo (Stop Line và Confirmation Line).

Output:
    - Optional[ViolationEvent]: Đối tượng sự kiện vi phạm chứa đầy đủ siêu dữ liệu
      phục vụ trích xuất bằng chứng (hoặc None nếu không vi phạm).
"""

from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    from src.traffic_light import LightState, TrafficLightResult
    from src.tracker import TrackedObject
except ImportError:
    from traffic_light import LightState, TrafficLightResult
    from tracker import TrackedObject

logger = logging.getLogger("ViolationLogic")


@dataclass
class ViolationEvent:
    """
    Cấu trúc đại diện cho một sự kiện vi phạm vượt đèn đỏ đã được xác nhận.

    Attributes:
        violation_id (str): Mã định danh duy nhất (UUID ngắn 8 ký tự).
        track_id (int): ID của phương tiện theo dõi vi phạm.
        vehicle_class (str): Tên loại phương tiện (car, motorcycle, bus, truck).
        timestamp (float): Mốc thời gian Unix (giây) khi vi phạm xảy ra.
        frame_index (int): Chỉ số khung hình video tại thời điểm vượt vạch xác nhận.
        status (str): Trạng thái sự kiện ("CONFIRMED").
        stop_line_frame (int): Khung hình khi xe cắt vạch dừng (Vạch 1).
        confirm_line_frame (int): Khung hình khi xe cắt vạch xác nhận (Vạch 2).
        evidence_path (Optional[str]): Đường dẫn thư mục lưu ảnh/clip bằng chứng.
    """
    violation_id: str
    track_id: int
    vehicle_class: str
    timestamp: float
    frame_index: int
    status: str = "CONFIRMED"
    stop_line_frame: int = 0
    confirm_line_frame: int = 0
    evidence_path: Optional[str] = None


def cross_product_2d(
    o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]
) -> float:
    """
    Tính tích có hướng (2D Cross Product) của 2 vector OA và OB:
        OA x OB = (Ax - Ox) * (By - Oy) - (Ay - Oy) * (Bx - Ox)

    Dấu của tích có hướng xác định vị trí tương đối của điểm B so với đường thẳng OA.
    """
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def segments_intersect(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    p3: Tuple[int, int],
    p4: Tuple[int, int],
) -> bool:
    """
    Kiểm tra sự giao nhau giữa 2 đoạn thẳng P1P2 (vạch kẻ) và P3P4 (bước di chuyển xe).

    Sử dụng giải thuật kiểm tra chiều quay (orientation test) với tích có hướng 2D.

    Args:
        p1, p2: Điểm đầu và điểm cuối của đoạn thẳng thứ nhất (vạch ảo).
        p3, p4: Điểm đầu và điểm cuối của đoạn thẳng thứ hai (bước di chuyển của xe).

    Returns:
        bool: True nếu hai đoạn thẳng cắt nhau, False nếu không.
    """
    d1 = cross_product_2d(p3, p4, p1)
    d2 = cross_product_2d(p3, p4, p2)
    d3 = cross_product_2d(p1, p2, p3)
    d4 = cross_product_2d(p1, p2, p4)

    # Nếu 2 đầu P1, P2 nằm về 2 phía của P3P4 và 2 đầu P3, P4 nằm về 2 phía của P1P2
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    # Trường hợp điểm nằm thẳng hàng trên đoạn (collinear check)
    # Ở dữ liệu video thực tế, xác suất điểm rời rạc rơi chính xác trên pixel vạch là rất nhỏ
    return False


class ViolationDetector:
    """
    Bộ suy luận logic vi phạm vượt đèn đỏ theo mô hình hai vạch ảo không - thời gian.

    Quy trình kiểm tra:
      1. Đèn giao thông phải giữ trạng thái ĐỎ liên tục ít nhất `red_stable_frames`.
      2. Quỹ đạo xe phải có độ dài ít nhất `min_track_len` điểm để tránh nhiễu khởi tạo.
      3. Phương tiện chưa từng bị kết luận vi phạm trước đó (chống trùng lặp sự kiện).
      4. Kiểm tra xe cắt qua Stop Line -> chuyển trạng thái xe sang CROSSED_STOP.
      5. Nếu xe đã cắt Stop Line và tiếp tục cắt qua Confirmation Line -> CONFIRMED VIOLATION!

    Args:
        stop_line (List[List[int]]): Tọa độ vạch dừng [[x1, y1], [x2, y2]].
        confirm_line (List[List[int]]): Tọa độ vạch xác nhận [[x1, y1], [x2, y2]].
        min_track_len (int): Số điểm quỹ đạo tối thiểu cần tích lũy.
        red_stable_frames (int): Số frame đèn đỏ ổn định tối thiểu trước khi bắt phạt.
        allow_right_turn (bool): Bật/tắt ngoại lệ cho phép rẽ phải khi đèn đỏ.
        target_classes (Optional[List[str]]): Danh sách lớp phương tiện áp dụng xử lý.
    """

    def __init__(
        self,
        stop_line: List[List[int]],
        confirm_line: List[List[int]],
        min_track_len: int = 10,
        red_stable_frames: int = 15,
        allow_right_turn: bool = False,
        target_classes: Optional[List[str]] = None,
    ) -> None:
        self.stop_line = stop_line
        self.confirm_line = confirm_line
        self.min_track_len = int(min_track_len)
        self.red_stable_frames = int(red_stable_frames)
        self.allow_right_turn = bool(allow_right_turn)
        self.target_classes = target_classes or ["car", "motorcycle", "bus", "truck"]

        # Trạng thái theo dõi thời gian thực
        self._red_frame_count: int = 0
        self._crossed_stop_info: Dict[int, int] = {}  # track_id -> frame_index cắt vạch 1
        self._confirmed_violations: Set[int] = set()  # track_id đã bị phạt

    def update_light(self, light_result: TrafficLightResult) -> None:
        """
        Cập nhật bộ đếm trạng thái đèn tín hiệu.

        Args:
            light_result: Kết quả nhận diện đèn tại frame hiện tại.
        """
        if light_result.state == LightState.RED:
            self._red_frame_count += 1
        else:
            self._red_frame_count = 0
            # Khi đèn chuyển sang Xanh/Vàng, xóa danh sách chờ xác nhận để không phạt oan
            self._crossed_stop_info.clear()

    @property
    def is_red_stable(self) -> bool:
        """Kiểm tra đèn đỏ đã duy trì đủ thời gian đệm ổn định hay chưa."""
        return self._red_frame_count >= self.red_stable_frames

    def check(
        self,
        track: TrackedObject,
        frame_index: int,
    ) -> Optional[ViolationEvent]:
        """
        Đánh giá trạng thái vi phạm cho một phương tiện đang theo dõi.

        Args:
            track: Đối tượng TrackedObject cần đánh giá.
            frame_index: Chỉ số khung hình hiện tại.

        Returns:
            Optional[ViolationEvent]: Sự kiện vi phạm nếu thỏa mãn đầy đủ các điều kiện.
        """
        # Điều kiện 0: Lọc đúng lớp phương tiện quy định
        if track.class_name not in self.target_classes:
            return None

        # Điều kiện 1: Đèn đỏ phải ổn định
        if not self.is_red_stable:
            return None

        # Điều kiện 2: Độ dài quỹ đạo phương tiện đủ lớn
        if len(track.history) < self.min_track_len:
            return None

        # Điều kiện 3: Không xử phạt lặp lại cùng một track_id
        if track.track_id in self._confirmed_violations:
            return None

        # Điểm đầu - cuối của 2 vạch ảo
        sl_p1 = (self.stop_line[0][0], self.stop_line[0][1])
        sl_p2 = (self.stop_line[1][0], self.stop_line[1][1])
        cl_p1 = (self.confirm_line[0][0], self.confirm_line[0][1])
        cl_p2 = (self.confirm_line[1][0], self.confirm_line[1][1])

        history_pts = list(track.history)

        # Bước A: Kiểm tra xem xe có cắt Vạch Dừng (Stop Line) không
        if track.track_id not in self._crossed_stop_info:
            for i in range(len(history_pts) - 1):
                p_prev = history_pts[i]
                p_curr = history_pts[i + 1]
                if segments_intersect(sl_p1, sl_p2, p_prev, p_curr):
                    self._crossed_stop_info[track.track_id] = frame_index
                    logger.debug(
                        f"Track #{track.track_id} ({track.class_name}) "
                        f"vừa cắt Vạch Dừng tại frame {frame_index}."
                    )
                    break

        # Bước B: Nếu xe đã từng cắt Vạch Dừng, kiểm tra tiếp Vạch Xác Nhận
        if track.track_id in self._crossed_stop_info:
            stop_frame = self._crossed_stop_info[track.track_id]

            for i in range(len(history_pts) - 1):
                p_prev = history_pts[i]
                p_curr = history_pts[i + 1]
                if segments_intersect(cl_p1, cl_p2, p_prev, p_curr):
                    # Kiểm tra ngoại lệ rẽ phải nếu được kích hoạt
                    if self.allow_right_turn and self._is_turning_right(history_pts):
                        logger.info(f"Bỏ qua vi phạm: Track #{track.track_id} rẽ phải hợp lệ.")
                        return None

                    # XÁC NHẬN VI PHẠM
                    self._confirmed_violations.add(track.track_id)
                    event_id = str(uuid.uuid4())[:8]

                    event = ViolationEvent(
                        violation_id=event_id,
                        track_id=track.track_id,
                        vehicle_class=track.class_name,
                        timestamp=time.time(),
                        frame_index=frame_index,
                        status="CONFIRMED",
                        stop_line_frame=stop_frame,
                        confirm_line_frame=frame_index,
                    )
                    logger.warning(
                        f"🚨 PHÁT HIỆN VI PHẠM: ID={event.violation_id} | "
                        f"Xe #{track.track_id} ({track.class_name}) "
                        f"tại frame {frame_index}!"
                    )
                    return event

        return None

    def _is_turning_right(self, history: List[Tuple[int, int]]) -> bool:
        """Ước tính hướng di chuyển xe xem có rẽ phải hay không."""
        if len(history) < 5:
            return False
        start_pt = history[0]
        end_pt = history[-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        # Nếu độ lệch sang phải (dx > 0) vượt trội so với đi thẳng
        return dx > 40 and abs(dx) > abs(dy) * 0.8

    def reset(self) -> None:
        """Reset toàn bộ bộ đếm và trạng thái vi phạm."""
        self._red_frame_count = 0
        self._crossed_stop_info.clear()
        self._confirmed_violations.clear()
