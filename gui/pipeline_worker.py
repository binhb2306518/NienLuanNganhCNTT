"""
gui/pipeline_worker.py - Luong xu ly AI (QThread)
===================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Chạy toàn bộ pipeline AI (YOLO detect → ByteTrack → ViolationDetector →
    EvidenceExtractor) trong một QThread riêng biệt, hoàn toàn tách rời luồng
    UI để đảm bảo giao diện không bị đơ trong quá trình suy luận.

Giao tiếp với MainWindow thông qua Qt Signals:
    - frame_ready   : Phát frame đã vẽ overlay (bytes JPEG) + FPS hiện tại.
    - violation_found: Phát ViolationEvent mới được xác nhận.
    - stats_updated : Phát dict thống kê {fps, tracks, frame_idx, total_frames}.
    - finished      : Phát khi video kết thúc hoặc dừng bởi người dùng.
    - error_occurred: Phát thông báo lỗi dạng str.
"""

from __future__ import annotations
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

# Đường dẫn gốc
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.violation_logic import ViolationDetector, ViolationEvent
from src.evidence import EvidenceExtractor
from src.traffic_light import LightState, TrafficLightResult, TrafficLightClassifier
from src.tracker import TrackedObject

logger = logging.getLogger("PipelineWorker")

# Màu sắc overlay (BGR)
_COLOR_NORMAL   = (0, 220, 0)
_COLOR_CROSSED  = (0, 165, 255)
_COLOR_VIOLATED = (0, 0, 255)
_COLOR_STOP     = (0, 0, 230)
_COLOR_CONFIRM  = (0, 215, 255)
_COLOR_RED_HUD  = (60, 60, 240)
_COLOR_WHITE    = (255, 255, 255)
_COLOR_BG       = (25, 25, 25)


class PipelineWorker(QThread):
    """
    QThread thực thi pipeline AI: YOLO + ByteTrack + ViolationDetector.

    Signals:
        frame_ready(bytes, float): JPEG bytes của frame đã annotate + FPS tức thời.
        violation_found(dict): Thông tin vi phạm dưới dạng dict tuần tự hóa.
        stats_updated(dict): Từ điển thống kê tổng hợp sau mỗi frame.
        finished(): Báo hiệu pipeline kết thúc.
        error_occurred(str): Thông báo lỗi.
    """

    frame_ready     = pyqtSignal(bytes, float)   # (jpeg_bytes, fps)
    violation_found = pyqtSignal(dict)           # serialized ViolationEvent
    stats_updated   = pyqtSignal(dict)           # {fps, tracks, frame_idx, total}
    finished        = pyqtSignal()
    error_occurred  = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        model_path: str,
        stop_line: List[List[int]],
        confirm_line: List[List[int]],
        evidence_dir: str = "evidence",
        conf_thresh: float = 0.35,
        img_size: int = 640,
        red_stable_frames: int = 15,
        allow_right_turn: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path       = video_path
        self.model_path       = model_path
        self.stop_line        = stop_line
        self.confirm_line     = confirm_line
        self.evidence_dir     = evidence_dir
        self.conf_thresh      = conf_thresh
        self.img_size         = img_size
        self.red_stable_frames = red_stable_frames
        self.allow_right_turn = allow_right_turn

        self._running  = True
        self._paused   = False
        self._mutex    = QMutex()

    # ------------------------------------------------------------------
    # Điều khiển từ luồng UI
    # ------------------------------------------------------------------
    def pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._paused = True

    def resume(self) -> None:
        with QMutexLocker(self._mutex):
            self._paused = False

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
            self._paused  = False

    # ------------------------------------------------------------------
    # Chuyển đổi kết quả YOLO → TrackedObject
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_results(
        results,
        track_states: Dict[int, TrackedObject],
        frame_idx: int,
        history_len: int = 50,
    ) -> List[TrackedObject]:
        if not results or results[0].boxes is None:
            return []
        boxes_data = results[0].boxes
        if boxes_data.id is None:
            return []

        CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "traffic_light"}
        active: List[TrackedObject] = []

        ids   = boxes_data.id.cpu().numpy().astype(int)
        xyxy  = boxes_data.xyxy.cpu().numpy().astype(int)
        confs = boxes_data.conf.cpu().numpy().astype(float)
        clses = boxes_data.cls.cpu().numpy().astype(int)

        for tid, box, conf, cls_id in zip(ids, xyxy, confs, clses):
            x1, y1, x2, y2 = box
            bc = ((x1 + x2) // 2, y2)
            cname = CLASS_NAMES.get(int(cls_id), f"cls{cls_id}")

            if tid not in track_states:
                track_states[tid] = TrackedObject(
                    track_id=int(tid), class_id=int(cls_id), class_name=cname,
                    confidence=float(conf), box=list(box), bottom_center=bc,
                    history=deque(maxlen=history_len), frame_count=0,
                    velocity=(0.0, 0.0), last_frame_idx=frame_idx,
                )
            else:
                obj = track_states[tid]
                if obj.history:
                    prev = obj.history[-1]
                    obj.velocity = (float(bc[0] - prev[0]), float(bc[1] - prev[1]))
                obj.box = list(box)
                obj.bottom_center = bc
                obj.confidence = float(conf)
                obj.frame_count += 1
                obj.last_frame_idx = frame_idx

            track_states[tid].history.append(bc)
            active.append(track_states[tid])
        return active

    # ------------------------------------------------------------------
    # Ve overlay lên frame
    # ------------------------------------------------------------------
    def _draw_overlay(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        violated_ids: set,
        crossed_ids: set,
        light_state: LightState,
        fps: float,
        frame_idx: int,
        total_frames: int,
        violation_count: int,
    ) -> np.ndarray:
        out = frame.copy()

        # Vạch dừng
        sl = self.stop_line
        cl = self.confirm_line
        cv2.line(out, (sl[0][0], sl[0][1]), (sl[1][0], sl[1][1]), _COLOR_STOP, 2)
        cv2.putText(out, "STOP LINE", (sl[0][0] + 5, sl[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_STOP, 2)
        cv2.line(out, (cl[0][0], cl[0][1]), (cl[1][0], cl[1][1]), _COLOR_CONFIRM, 2)
        cv2.putText(out, "CONFIRM LINE", (cl[0][0] + 5, cl[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_CONFIRM, 2)

        # Bounding boxes
        for t in tracks:
            x1, y1, x2, y2 = t.box
            if t.track_id in violated_ids:
                color, label = _COLOR_VIOLATED, f"[VI PHAM] #{t.track_id} {t.class_name}"
                thick = 3
            elif t.track_id in crossed_ids:
                color, label = _COLOR_CROSSED, f"[?] #{t.track_id} {t.class_name}"
                thick = 2
            else:
                color = _COLOR_NORMAL
                label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
                thick = 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(out, (x1, ly - th - 2), (x1 + tw + 4, ly + 2), color, -1)
            cv2.putText(out, label, (x1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_WHITE, 1)
            # Vết quỹ đạo
            pts = list(t.history)
            for i in range(1, len(pts)):
                cv2.circle(out, pts[i], 2, color, -1)

        # HUD góc trên trái
        hud = [
            f"Frame: {frame_idx}/{total_frames}",
            f"FPS:   {fps:.1f}",
            f"Tracks:{len(tracks):3d}",
            f"VI PHAM:{violation_count:3d}",
            f"Den:  {light_state.value}",
        ]
        overlay = out.copy()
        cv2.rectangle(overlay, (5, 5), (195, 115), _COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        for i, txt in enumerate(hud):
            col = _COLOR_RED_HUD if ("VI PHAM" in txt and violation_count > 0) else _COLOR_WHITE
            cv2.putText(out, txt, (10, 24 + i * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        # Đèn giao thông góc trên phải
        h, w = out.shape[:2]
        lt_color = {
            LightState.RED: (0, 0, 220),
            LightState.GREEN: (0, 200, 0),
            LightState.YELLOW: (0, 200, 220),
            LightState.UNKNOWN: (120, 120, 120),
        }.get(light_state, (120, 120, 120))
        cv2.circle(out, (w - 30, 30), 20, lt_color, -1)
        cv2.circle(out, (w - 30, 30), 20, _COLOR_WHITE, 2)
        cv2.putText(out, light_state.value, (w - 70, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lt_color, 2)

        return out

    # ------------------------------------------------------------------
    # Luồng chính
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            import torch
            from ultralytics import YOLO

            # Tải model
            model  = YOLO(self.model_path)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device == "cuda:0":
                model.model.half()
            logger.info(f"Model tải thành công trên {device.upper()}")

            # Mở video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Không mở được video: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Khởi tạo các module
            vio_det = ViolationDetector(
                stop_line=self.stop_line,
                confirm_line=self.confirm_line,
                min_track_len=10,
                red_stable_frames=self.red_stable_frames,
                allow_right_turn=self.allow_right_turn,
            )
            extractor = EvidenceExtractor(
                output_dir=self.evidence_dir,
                pre_frames=45,
                post_frames=30,
                fps=src_fps,
            )
            light_cls = TrafficLightClassifier(roi=[0.0, 0.0, 0.3, 0.3])

            track_states: Dict[int, TrackedObject] = {}
            violated_ids: set = set()
            frame_idx = 0
            fps_timer = time.perf_counter()
            fps_val   = 0.0
            FPS_WIN   = 30

            while self._running:
                # Kiểm tra tạm dừng
                if self._paused:
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                t0 = time.perf_counter()

                # YOLO tracking
                results = model.track(
                    source=frame, conf=self.conf_thresh,
                    imgsz=self.img_size, tracker="bytetrack.yaml",
                    persist=True, verbose=False, device=device,
                    classes=[0, 1, 2, 3],
                )

                tracks = self._parse_results(results, track_states, frame_idx)

                # Nhận diện đèn tín hiệu (HSV)
                light_res = light_cls.classify(frame)

                vio_det.update_light(light_res)
                extractor.push_frame(frame_idx, frame)

                # Kiểm tra vi phạm
                for t in tracks:
                    ev = vio_det.check(t, frame_idx)
                    if ev is not None:
                        violated_ids.add(t.track_id)
                        try:
                            ev_path = extractor.save(ev, crossing_frame=frame)
                        except Exception:
                            ev_path = ""
                        # Phát tín hiệu vi phạm (dict để tuần tự hóa qua signal)
                        self.violation_found.emit({
                            "violation_id"    : ev.violation_id,
                            "track_id"        : ev.track_id,
                            "vehicle_class"   : ev.vehicle_class,
                            "timestamp"       : ev.timestamp,
                            "frame_index"     : ev.frame_index,
                            "stop_line_frame" : ev.stop_line_frame,
                            "confirm_line_frame": ev.confirm_line_frame,
                            "evidence_path"   : ev_path,
                        })

                # Tính FPS
                if frame_idx % FPS_WIN == 0 and frame_idx > 0:
                    fps_val = FPS_WIN / max(time.perf_counter() - fps_timer, 1e-9)
                    fps_timer = time.perf_counter()

                # Ve overlay
                crossed_ids = set(vio_det._crossed_stop_info.keys())
                annotated = self._draw_overlay(
                    frame, tracks, violated_ids, crossed_ids,
                    light_res.state, fps_val, frame_idx, total_frames, len(violated_ids),
                )

                # Encode JPEG và phát ra signal
                ok_enc, jpeg_buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if ok_enc:
                    self.frame_ready.emit(bytes(jpeg_buf.tobytes()), fps_val)

                # Thống kê
                self.stats_updated.emit({
                    "fps"         : fps_val,
                    "tracks"      : len(tracks),
                    "frame_idx"   : frame_idx,
                    "total_frames": total_frames,
                    "violations"  : len(violated_ids),
                    "light"       : light_res.state.value,
                })

                frame_idx += 1

            cap.release()

        except Exception as e:
            logger.exception("Lỗi nghiêm trọng trong PipelineWorker")
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()

