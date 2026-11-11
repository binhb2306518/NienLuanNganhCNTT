"""
src/evidence.py - Trich xuat va quan ly file bang chung vi pham
================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Tự động trích xuất, đóng gói và lưu trữ bộ chứng cứ vi phạm hoàn chỉnh
    ra ổ đĩa D khi một sự kiện vi phạm được xác nhận:
      1. frame_before.jpg  : Ảnh phương tiện trước khi chạm vạch dừng (~1 giây).
      2. frame_crossing.jpg: Ảnh phương tiện tại khoảnh khắc cắt vạch dừng.
      3. frame_after.jpg   : Ảnh phương tiện khi đã vượt qua vạch xác nhận vào giao lộ.
      4. clip.mp4          : Đoạn video bằng chứng độ dài ~5 giây (bao gồm cả trước và sau).
      5. evidence_info.json: Tệp siêu dữ liệu JSON chuẩn hóa phục vụ tra cứu/phạt nguội.

Input:
    - ViolationEvent: Sự kiện vi phạm được xác nhận từ ViolationDetector.
    - Bộ đệm khung hình (Frame Circular Buffer).

Output:
    - Thư mục lưu trữ bằng chứng riêng biệt trên ổ đĩa D (`evidence/violation_YYYYMMDD_HHMMSS_ID`).
"""

from __future__ import annotations
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from src.violation_logic import ViolationEvent
except ImportError:
    from violation_logic import ViolationEvent

logger = logging.getLogger("Evidence")


class EvidenceExtractor:
    """
    Bộ trích xuất và quản lý chứng cứ vi phạm giao thông đa phương tiện.

    Sử dụng bộ đệm vòng tròn (Circular Buffer) để duy trì lịch sử khung hình gần nhất.
    Đảm bảo không tốn dung lượng RAM laptop bằng cách giới hạn kích thước buffer vừa đủ.

    Attributes:
        output_dir (Path): Thư mục gốc lưu trữ toàn bộ các gói chứng cứ.
        pre_frames (int): Số khung hình lưu trước sự kiện (mặc định: 75 ~ 2.5s @ 30fps).
        post_frames (int): Số khung hình lưu sau sự kiện (mặc định: 75 ~ 2.5s @ 30fps).
        fps (float): Tốc độ khung hình của video đầu vào để ghi clip MP4.
        jpeg_quality (int): Chất lượng nén ảnh JPG [1 - 100].
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "D:/NIENLUANNGANH/evidence",
        pre_frames: int = 75,
        post_frames: int = 75,
        fps: float = 30.0,
        jpeg_quality: int = 95,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.pre_frames = int(pre_frames)
        self.post_frames = int(post_frames)
        self.fps = float(fps) if fps > 0 else 30.0
        self.jpeg_quality = int(jpeg_quality)

        # Buffer vòng tròn lưu các frame gần nhất kèm chỉ số frame index
        # Kích thước tối đa: pre_frames * 2
        self._buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self.pre_frames * 2)

    def push_frame(self, frame_index: int, frame: np.ndarray) -> None:
        """
        Đẩy khung hình mới nhất vào bộ đệm vòng tròn.

        Args:
            frame_index: Chỉ số của khung hình.
            frame: Ảnh mảng numpy BGR kích thước (H, W, 3).
        """
        # Lưu bản copy nhẹ để tránh bị ghi đè bởi vòng lặp hiển thị UI
        self._buffer.append((frame_index, frame.copy()))

    def save(
        self,
        event: ViolationEvent,
        crossing_frame: Optional[np.ndarray] = None,
        post_frames_list: Optional[List[np.ndarray]] = None,
    ) -> str:
        """
        Đóng gói và lưu toàn bộ bộ chứng cứ ra thư mục riêng biệt.

        Args:
            event: Đối tượng ViolationEvent cần lưu chứng cứ.
            crossing_frame: Frame tại thời điểm cắt vạch (nếu có, ưu tiên sử dụng).
            post_frames_list: Danh sách các frame thu được sau thời điểm vi phạm.

        Returns:
            str: Đường dẫn tuyệt đối tới thư mục chứng cứ vừa tạo.
        """
        # Tạo tên thư mục theo mốc thời gian và mã vi phạm
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
        folder_name = f"violation_{ts_str}_{event.violation_id}"
        event_dir = self.output_dir / folder_name
        event_dir.mkdir(parents=True, exist_ok=True)

        buffer_list = list(self._buffer)
        buffer_len = len(buffer_list)

        # --- 1. Xác định frame_before.jpg ---
        # Lấy frame lùi về quá khứ khoảng 30 frame (~1 giây trước)
        if buffer_len > 0:
            before_idx = max(0, buffer_len - min(30, buffer_len))
            frame_before = buffer_list[before_idx][1]
        else:
            frame_before = crossing_frame if crossing_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # --- 2. Xác định frame_crossing.jpg ---
        if crossing_frame is not None:
            frame_cross = crossing_frame
        elif buffer_len > 0:
            frame_cross = buffer_list[-1][1]
        else:
            frame_cross = np.zeros((480, 640, 3), dtype=np.uint8)

        # --- 3. Xác định frame_after.jpg ---
        if post_frames_list and len(post_frames_list) > 0:
            after_idx = min(len(post_frames_list) - 1, 20)
            frame_after = post_frames_list[after_idx]
        elif buffer_len > 0:
            frame_after = buffer_list[-1][1]
        else:
            frame_after = frame_cross

        # Lưu 3 ảnh bằng chứng định dạng JPG chất lượng cao
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        path_before = event_dir / "frame_before.jpg"
        path_crossing = event_dir / "frame_crossing.jpg"
        path_after = event_dir / "frame_after.jpg"

        cv2.imwrite(str(path_before), frame_before, encode_params)
        cv2.imwrite(str(path_crossing), frame_cross, encode_params)
        cv2.imwrite(str(path_after), frame_after, encode_params)

        # --- 4. Tạo video clip MP4 minh họa vi phạm ---
        clip_frames: List[np.ndarray] = []
        # Lấy tối đa pre_frames từ buffer
        pre_slice = [f[1] for f in buffer_list[-self.pre_frames :]]
        clip_frames.extend(pre_slice)

        if post_frames_list:
            clip_frames.extend(post_frames_list[: self.post_frames])

        clip_path = event_dir / "clip.mp4"
        if clip_frames:
            h, w = clip_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (w, h))
            for f in clip_frames:
                writer.write(f)
            writer.release()
            logger.info(f"Đã xuất clip bằng chứng: {clip_path} ({len(clip_frames)} frames @ {self.fps}fps)")

        # --- 5. Lưu tệp siêu dữ liệu JSON ---
        info_data = {
            "violation_id": event.violation_id,
            "track_id": event.track_id,
            "vehicle_class": event.vehicle_class,
            "timestamp": event.timestamp,
            "datetime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(event.timestamp)),
            "frame_index": event.frame_index,
            "stop_line_frame": event.stop_line_frame,
            "confirm_line_frame": event.confirm_line_frame,
            "status": event.status,
            "files": {
                "frame_before": "frame_before.jpg",
                "frame_crossing": "frame_crossing.jpg",
                "frame_after": "frame_after.jpg",
                "clip": "clip.mp4" if clip_frames else None,
            },
        }

        json_path = event_dir / "evidence_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)

        # Gán ngược đường dẫn thư mục vào đối tượng event
        event.evidence_path = str(event_dir)
        logger.info(f"✅ Gói bằng chứng đã lưu hoàn tất: {event_dir}")
        return str(event_dir)

    def clear(self) -> None:
        """Xóa trắng bộ đệm khung hình."""
        self._buffer.clear()
