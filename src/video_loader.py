"""
src/video_loader.py - Module doc va chuan hoa luong video dau vao
=================================================================
Input:  Duong dan file MP4 hoac URL RTSP camera
Output: Frame numpy (H, W, 3) BGR, da resize chuan hoa
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional


class VideoLoader:
    """
    Wrapper cho OpenCV VideoCapture.
    Ho tro ca file video va camera RTSP.

    Args:
        source (str | int): Duong dan file video (.mp4) hoac RTSP URL hoac index camera (0).
        resize_width (int):  Chieu rong muc tieu sau resize. 0 = giu nguyen kich thuoc goc.
        fps_limit (float):   Gioi han FPS xu ly. 0 = khong gioi han.
    """

    def __init__(self, source: str | int, resize_width: int = 1280, fps_limit: float = 0.0):
        self.source = source
        self.resize_width = resize_width
        self.fps_limit = fps_limit

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_original: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.total_frames: int = 0

    def open(self) -> bool:
        """Mo ket noi den nguon video. Tra ve True neu thanh cong."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        self.fps_original  = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def release(self):
        """Giai phong tai nguyen VideoCapture."""
        if self.cap:
            self.cap.release()

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Doc mot frame tu nguon video.
        Returns:
            (success, frame): success la bool, frame la numpy BGR array.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        if self.resize_width > 0 and frame.shape[1] != self.resize_width:
            ratio  = self.resize_width / frame.shape[1]
            new_h  = int(frame.shape[0] * ratio)
            frame  = cv2.resize(frame, (self.resize_width, new_h), interpolation=cv2.INTER_AREA)
        return True, frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator yield tung frame lien tiep den het video."""
        if not self.open():
            raise RuntimeError(f"Khong mo duoc video: {self.source}")
        try:
            while True:
                ok, frame = self.read_frame()
                if not ok:
                    break
                yield frame
        finally:
            self.release()

    @property
    def scale_ratio(self) -> float:
        """Ti le scale so voi khung hinh goc."""
        if self.resize_width > 0 and self.frame_width > 0:
            return self.resize_width / self.frame_width
        return 1.0

