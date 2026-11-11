"""
src/traffic_light.py - Nhan dien trang thai den tin hieu giao thong
====================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Xác định chính xác trạng thái tín hiệu của đèn giao thông (ĐỎ, VÀNG, XANH)
    tại nút giao. Cung cấp 2 phương thức tiếp cận:
      1. Phân tích không gian màu HSV trên vùng quan sát ROI cố định (tối ưu, nhẹ).
      2. Suy luận đối tượng qua YOLO detector/classifier (linh hoạt với camera di chuyển).
    Tích hợp bộ lọc làm mịn thời gian (Temporal Majority Voting) để loại bỏ hiện tượng
    nhấp nháy do tần số quét màn trập camera ngoài trời.

Input:
    - Frame ảnh numpy.ndarray (H, W, 3) BGR.
    - Vùng ROI định dạng pixel (x1, y1, x2, y2) hoặc chuẩn hóa [0.0, 1.0].

Output:
    - TrafficLightResult: Chứa trạng thái (LightState), độ tin cậy và phân bố tỉ lệ màu.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger("TrafficLight")


class LightState(str, Enum):
    """Trạng thái tín hiệu đèn giao thông chuẩn."""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrafficLightResult:
    """
    Kết quả phân tích trạng thái đèn tín hiệu tại một khung hình.

    Attributes:
        state (LightState): Trạng thái xác định (RED, YELLOW, GREEN, UNKNOWN).
        confidence (float): Độ tin cậy ước lượng [0.0, 1.0].
        box (Optional[List[int]]): Tọa độ vùng đèn xét pixel [x1, y1, x2, y2].
        method (str): Phương pháp ("hsv" hoặc "yolo_cls").
        color_ratios (Dict[str, float]): Tỉ lệ diện tích màu phát hiện trong ROI.
    """
    state: LightState
    confidence: float
    box: Optional[List[int]] = None
    method: str = "hsv"
    color_ratios: Dict[str, float] = field(default_factory=dict)


class TrafficLightClassifier:
    """
    Bộ phân loại trạng thái đèn tín hiệu bằng giải thuật phân tích màu sắc HSV.

    Hỗ trợ:
      - 2 dải Hue cho màu Đỏ (đặc thù không gian hình trụ HSV bao quanh 0/180).
      - ROI chuẩn hóa [0.0, 1.0] tự động nhân theo độ phân giải frame đầu vào.
      - Bộ đệm lịch sử N frame để lọc nhiễu nhấp nháy đèn (flickering).

    Args:
        roi (Optional[Union[Tuple[int, int, int, int], List[float]]]):
            Tọa độ vùng đèn. Có thể là pixel [x1, y1, x2, y2] hoặc chuẩn hóa [0.0 - 1.0].
        min_ratio (float): Tỉ lệ diện tích pixel màu tối thiểu để xác nhận tín hiệu.
        smoothing_window (int): Kích thước cửa sổ trượt làm mịn kết quả thời gian.
    """

    # Ngưỡng màu HSV chuẩn (H: 0-180, S: 0-255, V: 0-255) trong OpenCV
    RED_RANGES = [
        (np.array([0, 120, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        (np.array([170, 120, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    ]
    YELLOW_RANGE = (
        np.array([15, 120, 70], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8),
    )
    GREEN_RANGE = (
        np.array([45, 100, 70], dtype=np.uint8),
        np.array([85, 255, 255], dtype=np.uint8),
    )

    def __init__(
        self,
        roi: Optional[Union[Tuple[int, int, int, int], List[float], Tuple[float, ...]]] = None,
        min_ratio: float = 0.10,
        smoothing_window: int = 5,
    ) -> None:
        self.roi = roi
        self.min_ratio = float(min_ratio)
        self.smoothing_window = max(1, int(smoothing_window))

        # Bộ đệm lưu trữ trạng thái gần nhất để làm mịn
        self._history_states: Deque[LightState] = deque(maxlen=self.smoothing_window)

    def _resolve_roi(self, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Chuyển đổi ROI từ tọa độ chuẩn hóa sang tọa độ pixel thực tế."""
        if self.roi is None:
            return None

        h, w = frame_shape[:2]
        r = self.roi

        # Kiểm tra nếu tọa độ là chuẩn hóa [0.0, 1.0]
        if all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in r):
            x1 = int(r[0] * w)
            y1 = int(r[1] * h)
            x2 = int(r[2] * w)
            y2 = int(r[3] * h)
            return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

        # Tọa độ pixel thông thường
        return (int(r[0]), int(r[1]), int(r[2]), int(r[3]))

    def classify(self, frame: np.ndarray) -> TrafficLightResult:
        """
        Phân loại trạng thái đèn tín hiệu từ khung hình.

        Args:
            frame: Ảnh numpy BGR kích thước (H, W, 3).

        Returns:
            TrafficLightResult: Kết quả phân tích kèm trạng thái đã được làm mịn.
        """
        pixel_roi = self._resolve_roi(frame.shape)

        if pixel_roi is not None:
            x1, y1, x2, y2 = pixel_roi
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"ROI không hợp lệ: {pixel_roi}")
                return TrafficLightResult(LightState.UNKNOWN, 0.0)
            roi_img = frame[y1:y2, x1:x2]
        else:
            roi_img = frame

        if roi_img.size == 0:
            return TrafficLightResult(LightState.UNKNOWN, 0.0)

        # Chuyển sang không gian màu HSV
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        total_pixels = float(hsv.shape[0] * hsv.shape[1])

        # Tạo mask lọc màu
        red_mask_1 = cv2.inRange(hsv, self.RED_RANGES[0][0], self.RED_RANGES[0][1])
        red_mask_2 = cv2.inRange(hsv, self.RED_RANGES[1][0], self.RED_RANGES[1][1])
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        yellow_mask = cv2.inRange(hsv, self.YELLOW_RANGE[0], self.YELLOW_RANGE[1])
        green_mask = cv2.inRange(hsv, self.GREEN_RANGE[0], self.GREEN_RANGE[1])

        red_ratio = cv2.countNonZero(red_mask) / total_pixels
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
        green_ratio = cv2.countNonZero(green_mask) / total_pixels

        ratios = {
            LightState.RED: red_ratio,
            LightState.YELLOW: yellow_ratio,
            LightState.GREEN: green_ratio,
        }

        # Tìm màu có tỉ lệ cao nhất
        best_state, best_ratio = max(ratios.items(), key=lambda item: item[1])

        # Đánh giá theo ngưỡng tối thiểu
        if best_ratio < self.min_ratio:
            raw_state = LightState.UNKNOWN
            raw_conf = 0.0
        else:
            raw_state = best_state
            raw_conf = min(best_ratio / 0.35, 1.0)

        # Cập nhật bộ đệm lịch sử
        self._history_states.append(raw_state)

        # Lấy trạng thái chiếm đa số (Majority Voting) trong cửa sổ trượt
        smoothed_state = self._get_majority_state()

        return TrafficLightResult(
            state=smoothed_state,
            confidence=raw_conf,
            box=list(pixel_roi) if pixel_roi else None,
            method="hsv",
            color_ratios={
                "red": round(red_ratio, 4),
                "yellow": round(yellow_ratio, 4),
                "green": round(green_ratio, 4),
            },
        )

    def _get_majority_state(self) -> LightState:
        """Thực hiện bầu cử đa số trên cửa sổ trượt gần nhất."""
        if not self._history_states:
            return LightState.UNKNOWN

        counts: Dict[LightState, int] = {}
        for s in self._history_states:
            counts[s] = counts.get(s, 0) + 1

        majority_state, _ = max(counts.items(), key=lambda item: item[1])
        return majority_state

    def reset(self) -> None:
        """Reset bộ đệm làm mịn trạng thái."""
        self._history_states.clear()
