"""
src/evidence.py - Trich xuat va luu bang chung vi pham
=======================================================
Khi phat hien vi pham, tu dong luu:
  - frame_before.jpg  : Frame truoc thoi diem cat vach
  - frame_crossing.jpg: Frame tai thoi diem cat vach
  - frame_after.jpg   : Frame sau khi xe da vao giao lo
  - clip.mp4          : Video clip ngan 5 giay
  - evidence_info.json: Thong tin su kien (JSON)

Input:  ViolationEvent + buffer frame
Output: Thu muc bang chung tren o dia D
"""
from __future__ import annotations
import cv2
import json
import time
from pathlib import Path
from collections import deque
from typing import Optional
import numpy as np

from violation_logic import ViolationEvent


class EvidenceExtractor:
    """
    Luu bang chung vi pham ra o dia.

    Args:
        output_dir  (str) : Thu muc goc luu bang chung.
        pre_frames  (int) : So frame giu truoc su kien.
        post_frames (int) : So frame giu sau su kien de tao clip.
        fps         (float): FPS cua video dau vao.
    """

    def __init__(
        self,
        output_dir:  str   = "D:/NIENLUANNGANH/evidence",
        pre_frames:  int   = 75,
        post_frames: int   = 75,
        fps:         float = 30.0,
    ):
        self.output_dir  = Path(output_dir)
        self.pre_frames  = pre_frames
        self.post_frames = post_frames
        self.fps         = fps

        # Buffer luu cac frame gan day
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=pre_frames * 2)

    def push_frame(self, frame: np.ndarray):
        """
        Day frame moi nhat vao buffer.
        Goi sau moi lan xu ly frame trong pipeline chinh.
        """
        self._frame_buffer.append(frame.copy())

    def save(
        self,
        event:         ViolationEvent,
        crossing_frame: np.ndarray,
        post_buffer:   list[np.ndarray],
    ) -> str:
        """
        Luu toan bo bang chung ra o dia.

        Args:
            event         : Su kien vi pham.
            crossing_frame: Frame tai thoi diem cat vach.
            post_buffer   : Danh sach frame sau su kien.
        Returns:
            Duong dan thu muc bang chung.
        """
        # Tao thu muc rieng cho moi su kien
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        folder_name   = f"violation_{timestamp_str}_{event.violation_id}"
        folder        = self.output_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        # --- 3 frame anh bang chung ---
        pre_frames_list = list(self._frame_buffer)
        if len(pre_frames_list) >= 3:
            before_idx = max(0, len(pre_frames_list) - 30)  # ~1 giay truoc
            frame_before = pre_frames_list[before_idx]
        else:
            frame_before = crossing_frame

        cv2.imwrite(str(folder / "frame_before.jpg"),   frame_before,   [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(folder / "frame_crossing.jpg"), crossing_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if post_buffer:
            after_idx    = min(len(post_buffer) - 1, 30)  # ~1 giay sau
            frame_after  = post_buffer[after_idx]
            cv2.imwrite(str(folder / "frame_after.jpg"), frame_after, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # --- Video clip MP4 ---
        clip_frames = list(pre_frames_list[-self.pre_frames:]) + [crossing_frame] + post_buffer[:self.post_frames]
        if clip_frames:
            h, w = clip_frames[0].shape[:2]
            fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
            writer  = cv2.VideoWriter(str(folder / "clip.mp4"), fourcc, self.fps, (w, h))
            for f in clip_frames:
                writer.write(f)
            writer.release()

        # --- File JSON thong tin ---
        info = {
            "violation_id":  event.violation_id,
            "track_id":      event.track_id,
            "vehicle_class": event.vehicle_class,
            "timestamp":     event.timestamp,
            "frame_index":   event.frame_index,
            "status":        event.status,
            "saved_at":      timestamp_str,
        }
        with open(folder / "evidence_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        event.evidence_path = str(folder)
        return str(folder)

