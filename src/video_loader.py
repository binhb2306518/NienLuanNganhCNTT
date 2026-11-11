"""
src/video_loader.py - Module doc va chuan hoa luong video dau vao
=================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Đọc, giải mã và chuẩn hóa các luồng video từ tập tin (.mp4, .avi, .mkv)
    hoặc URL RTSP từ camera giao thông thông minh. Hỗ trợ resize tỉ lệ giữ nguyên
    aspect ratio, giới hạn FPS, và cơ chế Context Manager an toàn.

Input:
    - source (str | int): Đường dẫn tập tin video, luồng RTSP camera hoặc ID webcam.
    - resize_width (int): Chiều rộng chuẩn hóa (pixel). Mặc định 1280. 0 = giữ nguyên.
    - fps_limit (float):  Giới hạn FPS xử lý để tối ưu tài nguyên laptop (0 = unconstrained).

Output:
    - Frame ảnh BGR dạng numpy.ndarray kích thước (H, W, 3), dtype=uint8.
    - Generator luồng frame lien tục phục vụ vòng lặp xử lý chính.
"""

from __future__ import annotations
import cv2
import time
import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("VideoLoader")


class VideoLoader:
    """
    Bộ nạp và chuẩn hóa luồng video phục vụ pipeline thị giác máy tính.

    Hỗ trợ đọc từ video offline (mp4, avi) và camera trực tiếp (RTSP/USB).
    Tích hợp context manager (`with VideoLoader(...) as loader:`) để tự động giải phóng bộ nhớ.

    Attributes:
        source (str | int): Nguồn video (file path, rtsp://..., hoặc webcam index).
        resize_width (int): Chiều rộng sau chuẩn hóa (pixel).
        fps_limit (float): Tốc độ khung hình tối đa muốn xử lý.
        cap (Optional[cv2.VideoCapture]): Đối tượng VideoCapture của OpenCV.
        fps_original (float): Tốc độ khung hình gốc của video.
        frame_width (int): Chiều rộng khung hình gốc.
        frame_height (int): Chiều cao khung hình gốc.
        total_frames (int): Tổng số khung hình của video (offline).
    """

    def __init__(
        self,
        source: Union[str, int, Path],
        resize_width: int = 1280,
        fps_limit: float = 0.0,
    ) -> None:
        """
        Khởi tạo VideoLoader với tham số chuẩn hóa.

        Args:
            source: Đường dẫn file video hoặc URI RTSP camera.
            resize_width: Chiều rộng ảnh chuẩn hóa (mặc định: 1280px). 0 để giữ nguyên.
            fps_limit: Giới hạn FPS đọc (0 = đọc theo tốc độ tối đa của phần cứng).
        """
        self.source: Union[str, int] = str(source) if isinstance(source, Path) else source
        self.resize_width: int = int(resize_width)
        self.fps_limit: float = float(fps_limit)

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_original: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.total_frames: int = 0
        self._last_frame_time: float = 0.0

    def open(self) -> bool:
        """
        Mở kết nối tới nguồn video và truy xuất siêu dữ liệu (metadata).

        Returns:
            bool: True nếu mở thành công, False nếu thất bại.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"Không thể mở nguồn video: {self.source}")
                return False

            fps_raw = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps_original = float(fps_raw) if fps_raw and fps_raw > 0 else 30.0
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(
                f"Đã mở video: {self.source} | "
                f"Kích thước gốc: {self.frame_width}x{self.frame_height} | "
                f"FPS: {self.fps_original:.2f} | Tổng frame: {self.total_frames}"
            )
            return True
        except Exception as e:
            logger.error(f"Ngoại lệ khi mở video: {e}")
            return False

    def release(self) -> None:
        """Giải phóng tài nguyên OpenCV VideoCapture an toàn."""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
            logger.info("Đã giải phóng tài nguyên VideoCapture.")

    def __enter__(self) -> VideoLoader:
        """Hỗ trợ Context Manager."""
        if not self.open():
            raise RuntimeError(f"Không thể mở nguồn video qua context manager: {self.source}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Tự động đóng VideoCapture khi thoát khối with."""
        self.release()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc một frame ảnh kế tiếp và thực hiện chuẩn hóa tỉ lệ.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - bool: True nếu đọc frame thành công.
                - np.ndarray: Mảng ảnh numpy 3 chiều (BGR) hoặc None nếu kết thúc/lỗi.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        # Điều tiết tốc độ khung hình (FPS limit) nếu được cấu hình
        if self.fps_limit > 0:
            interval = 1.0 / self.fps_limit
            elapsed = time.time() - self._last_frame_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_frame_time = time.time()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Resize chuẩn hóa chiều rộng nếu cần (giữ nguyên aspect ratio)
        if self.resize_width > 0 and frame.shape[1] != self.resize_width:
            ratio = self.resize_width / float(frame.shape[1])
            new_h = int(frame.shape[0] * ratio)
            frame = cv2.resize(frame, (self.resize_width, new_h), interpolation=cv2.INTER_AREA)

        return True, frame

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator sinh cặp (frame_index, frame_image) liên tục đến hết video.

        Yields:
            Tuple[int, np.ndarray]: Chỉ số khung hình (bắt đầu từ 0) và dữ liệu ảnh BGR.
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError(f"Không thể mở luồng video: {self.source}")

        frame_idx = 0
        try:
            while True:
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    break
                yield frame_idx, frame
                frame_idx += 1
        finally:
            self.release()

    @property
    def scale_ratio(self) -> float:
        """Tỉ lệ co dãn giữa chiều rộng chuẩn hóa và chiều rộng ảnh gốc."""
        if self.resize_width > 0 and self.frame_width > 0:
            return float(self.resize_width) / float(self.frame_width)
        return 1.0

    @property
    def duration_seconds(self) -> float:
        """Thời lượng ước tính của video tính bằng giây."""
        if self.fps_original > 0 and self.total_frames > 0:
            return self.total_frames / self.fps_original
        return 0.0
