"""
test_violation.py - Script kiem thu Logic Phat hien Vi pham Vuot Den Do
=======================================================================
Du an: He thong Tu dong Phat hien Phuong tien Vi pham Den Tin hieu Giao thong
Sinh vien: Nguyen Thanh Binh - B2306518 - CT466E - DH Can Tho

Muc dich:
    Chay thu toan bo pipeline phat hien vi pham tren video test 30s, bao gom:
      1. Doc video dau vao va chay YOLO + ByteTrack de lay TrackedObject.
      2. Gia lap deu tin hieu giao thong o trang thai DO on dinh (testing mode).
      3. Truyen tung TrackedObject qua ViolationDetector de kiem tra vi pham.
      4. Truyen frame vao EvidenceExtractor va luu bang chung khi co vi pham.
      5. Xuat video ket qua voi overlay cac vach ao, bounding box mau do/xanh.
      6. In ra man hinh tong ket so luong vi pham phat hien duoc.

Tieu chi nghiem thu (GD6):
    - Phan biet ro rang xe vuot ca hai vach (VI PHAM) voi xe chi cham vach 1 roi dung.
    - Khong co loi Python nao xay ra trong qua trinh chay.
    - Xuat duoc file evidence/ va test_violation_output.mp4.

Su dung:
    python test_violation.py [--video PATH] [--model PATH] [--output PATH]

Ket qua mau:
    [frame  45] Track #3 (car)  → CAT VACH DUNG
    [frame  52] Track #3 (car)  → VI PHAM XAC NHAN  |  ID: a1b2c3d4
    Tong ket: 2 vi pham duoc ghi nhan.
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Duong dan goc du an
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import cac module du an
# ---------------------------------------------------------------------------
try:
    from src.violation_logic import ViolationDetector, ViolationEvent
    from src.evidence import EvidenceExtractor
    from src.traffic_light import LightState, TrafficLightResult
    from src.tracker import TrackedObject
except ImportError as e:
    print(f"[LOI] Khong the import module: {e}")
    print("       Chay: pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestViolation")

# ---------------------------------------------------------------------------
# Hang so mac dinh
# ---------------------------------------------------------------------------
DEFAULT_VIDEO = str(PROJECT_ROOT / "dataset" / "test_30s_benchmark.mp4")
DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "pretrained" / "yolov8s.pt")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "test_violation_output.mp4")
DEFAULT_EVIDENCE = str(PROJECT_ROOT / "evidence")

STOP_LINE_COORDS    = [[100, 400], [900, 400]]   # Vach dung (Do)
CONFIRM_LINE_COORDS = [[100, 450], [900, 450]]   # Vach xac nhan (Vang)

# Cau hinh YOLO
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
IMG_SIZE    = 640
TRACKER_CFG = "bytetrack.yaml"

# Mau sac overlay (BGR)
COLOR_STOP_LINE    = (0, 0, 255)      # Do
COLOR_CONFIRM_LINE = (0, 215, 255)    # Vang
COLOR_NORMAL       = (0, 220, 0)      # Xanh la
COLOR_CROSSED_STOP = (0, 165, 255)    # Cam (da cat vach 1, chua xac nhan)
COLOR_VIOLATION    = (0, 0, 255)      # Do (da bi phat)
COLOR_HUD_BG       = (20, 20, 20)     # Nen HUD
COLOR_TEXT_WHITE   = (255, 255, 255)  # Chu trang
COLOR_TEXT_RED     = (80, 80, 255)    # Chu do


# ===========================================================================
# Ham tien ich
# ===========================================================================

def draw_virtual_lines(frame: np.ndarray) -> np.ndarray:
    """Ve 2 vach ao (Stop Line va Confirmation Line) len frame."""
    sl = STOP_LINE_COORDS
    cl = CONFIRM_LINE_COORDS

    # Vach dung - Do
    cv2.line(frame, (sl[0][0], sl[0][1]), (sl[1][0], sl[1][1]),
             COLOR_STOP_LINE, 2)
    cv2.putText(frame, "STOP LINE", (sl[0][0] + 5, sl[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_STOP_LINE, 2)

    # Vach xac nhan - Vang
    cv2.line(frame, (cl[0][0], cl[0][1]), (cl[1][0], cl[1][1]),
             COLOR_CONFIRM_LINE, 2)
    cv2.putText(frame, "CONFIRM LINE", (cl[0][0] + 5, cl[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CONFIRM_LINE, 2)

    return frame


def draw_track(
    frame: np.ndarray,
    track: TrackedObject,
    is_violated: bool,
    has_crossed_stop: bool,
) -> np.ndarray:
    """Ve bounding box va nhan len frame voi mau phan biet trang thai."""
    x1, y1, x2, y2 = track.box

    if is_violated:
        color = COLOR_VIOLATION
        label = f"[VI PHAM] #{track.track_id} {track.class_name}"
    elif has_crossed_stop:
        color = COLOR_CROSSED_STOP
        label = f"[NGHI VAN] #{track.track_id} {track.class_name}"
    else:
        color = COLOR_NORMAL
        label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"

    # Hop bounding box
    thickness = 3 if is_violated else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Nhan
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lbl_y = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, lbl_y - th - 2), (x1 + tw + 4, lbl_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 2, lbl_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Ve vet quy dao (bottom center history)
    pts = list(track.history)
    for i in range(1, len(pts)):
        cv2.circle(frame, pts[i], 2, color, -1)

    return frame


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    total_tracks: int,
    violation_count: int,
) -> np.ndarray:
    """Ve thanh HUD thong tin o goc tren trai."""
    h, w = frame.shape[:2]
    hud_lines = [
        f"Frame: {frame_idx:5d}",
        f"FPS:   {fps:5.1f}",
        f"Tracks: {total_tracks:3d}",
        f"VI PHAM: {violation_count:3d}",
        "Den: DO [TESTING]",
    ]

    # Nen ban trong
    padding = 8
    line_h = 20
    box_h = len(hud_lines) * line_h + padding * 2
    box_w = 185
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, text in enumerate(hud_lines):
        color = COLOR_TEXT_RED if "VI PHAM" in text and violation_count > 0 else COLOR_TEXT_WHITE
        cv2.putText(
            frame, text,
            (10, padding + (i + 1) * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
        )

    return frame


def parse_yolo_results(
    results,
    track_states: Dict[int, TrackedObject],
    frame_idx: int,
    history_len: int = 50,
) -> List[TrackedObject]:
    """
    Chuyen doi ket qua YOLO model.track() sang List[TrackedObject].

    Args:
        results: Danh sach ket qua YOLO (1 phan tu tuong ung 1 frame).
        track_states: Dict duy tri trang thai TrackedObject theo track_id.
        frame_idx: Chi so frame hien tai.
        history_len: Do dai toi da cua vet quy dao.

    Returns:
        Danh sach TrackedObject cap nhat cho frame hien tai.
    """
    if not results or results[0].boxes is None:
        return []

    boxes_data = results[0].boxes
    if boxes_data.id is None:
        return []

    # CLASS_NAMES khop voi dataset giao thong 5-class
    CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "traffic_light"}

    active_tracks: List[TrackedObject] = []
    ids   = boxes_data.id.cpu().numpy().astype(int)
    xyxy  = boxes_data.xyxy.cpu().numpy().astype(int)
    confs = boxes_data.conf.cpu().numpy().astype(float)
    clses = boxes_data.cls.cpu().numpy().astype(int)

    for tid, box, conf, cls_id in zip(ids, xyxy, confs, clses):
        x1, y1, x2, y2 = box
        bottom_center = ((x1 + x2) // 2, y2)
        cls_name = CLASS_NAMES.get(int(cls_id), f"cls{cls_id}")

        if tid not in track_states:
            # Khoi tao TrackedObject moi
            track_states[tid] = TrackedObject(
                track_id=int(tid),
                class_id=int(cls_id),
                class_name=cls_name,
                confidence=float(conf),
                box=list(box),
                bottom_center=bottom_center,
                history=deque(maxlen=history_len),
                frame_count=0,
                velocity=(0.0, 0.0),
                last_frame_idx=frame_idx,
            )
        else:
            obj = track_states[tid]
            # Tinh vector van toc
            if obj.history:
                prev = obj.history[-1]
                obj.velocity = (
                    float(bottom_center[0] - prev[0]),
                    float(bottom_center[1] - prev[1]),
                )
            obj.box = list(box)
            obj.bottom_center = bottom_center
            obj.confidence = float(conf)
            obj.frame_count += 1
            obj.last_frame_idx = frame_idx

        track_states[tid].history.append(bottom_center)
        active_tracks.append(track_states[tid])

    return active_tracks


# ===========================================================================
# Ham chinh
# ===========================================================================

def run_test(
    video_path: str,
    model_path: str,
    output_path: str,
    evidence_dir: str,
) -> None:
    """
    Chay toan bo pipeline kiem thu vi pham tren video dau vao.

    Args:
        video_path:   Duong dan video dau vao.
        model_path:   Duong dan file .pt cua model YOLO.
        output_path:  Duong dan file video ket qua.
        evidence_dir: Thu muc luu bang chung vi pham.
    """
    logger.info("=" * 60)
    logger.info("KIEM THU LOGIC PHAT HIEN VI PHAM (GD6)")
    logger.info("=" * 60)
    logger.info(f"Video dau vao : {video_path}")
    logger.info(f"Model         : {model_path}")
    logger.info(f"Video xuat    : {output_path}")
    logger.info(f"Bang chung    : {evidence_dir}")

    # ------------------------------------------------------------------
    # 1. Kiem tra dau vao
    # ------------------------------------------------------------------
    if not Path(video_path).exists():
        logger.error(f"Khong tim thay video: {video_path}")
        logger.info("Tao video test tu dau (synthetic)...")
        video_path = _create_synthetic_video(video_path)

    if not Path(model_path).exists():
        logger.warning(f"Khong tim thay model tai {model_path}")
        # Thu model yolo11s.pt
        alt = str(PROJECT_ROOT / "models" / "pretrained" / "yolo11s.pt")
        if Path(alt).exists():
            model_path = alt
            logger.info(f"Dung model thay the: {model_path}")
        else:
            logger.error("Khong tim thay bat ky model YOLO nao. Dung lai.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Khoi tao YOLO
    # ------------------------------------------------------------------
    logger.info("Dang tai model YOLO...")
    try:
        from ultralytics import YOLO
        import torch
        model = YOLO(model_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cuda:0":
            model.model.half()  # FP16 de tiet kiem VRAM
        logger.info(f"Model da san sang tren {device.upper()}.")
    except Exception as e:
        logger.error(f"Loi tai model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Khoi tao cac module
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Khong mo duoc video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {width}x{height} @ {src_fps:.1f}fps | {total_frames} frames")

    # ViolationDetector: 2 vach ao, can den do on dinh 15 frame
    detector_vio = ViolationDetector(
        stop_line=STOP_LINE_COORDS,
        confirm_line=CONFIRM_LINE_COORDS,
        min_track_len=10,
        red_stable_frames=15,
        allow_right_turn=False,
        target_classes=["car", "motorcycle", "bus", "truck"],
    )

    # Gia lap den do on dinh nhu testing mode
    # Cho den do len ngay tu frame 1 (goi update_light 30 lan de _red_frame_count >= red_stable_frames)
    fake_red = TrafficLightResult(state=LightState.RED, confidence=0.99)
    for _ in range(30):
        detector_vio.update_light(fake_red)    # EvidenceExtractor: luu bang chung vao thu muc evidence/
    extractor = EvidenceExtractor(
        output_dir=evidence_dir,
        pre_frames=45,
        post_frames=30,
        fps=src_fps,
        jpeg_quality=92,
    )

    # VideoWriter de xuat video ket qua
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))
    if not out_writer.isOpened():
        logger.warning(f"Khong the mo VideoWriter cho {output_path}. Bo qua xuat video.")
        out_writer = None

    # ------------------------------------------------------------------
    # 4. Vong lap chinh
    # ------------------------------------------------------------------
    track_states: Dict[int, TrackedObject] = {}   # Luu trang thai TrackedObject
    violated_ids: set = set()                     # tap track_id da bi phat
    all_violations: List[ViolationEvent] = []

    frame_idx    = 0
    fps_timer    = time.perf_counter()
    fps_display  = 0.0
    fps_window   = 30  # Tinh FPS trung binh 30 frame

    logger.info("Bat dau xu ly video...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_start = time.perf_counter()

        # ----------------------------------------------------------------
        # 4a. YOLO tracking
        # ----------------------------------------------------------------
        results = model.track(
            source=frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            tracker=TRACKER_CFG,
            persist=True,
            verbose=False,
            device=device,
            classes=[0, 1, 2, 3],   # car, motorcycle, bus, truck (bo traffic_light)
        )

        # ----------------------------------------------------------------
        # 4b. Chuyen doi ket qua sang TrackedObject
        # ----------------------------------------------------------------
        active_tracks = parse_yolo_results(results, track_states, frame_idx)

        # ----------------------------------------------------------------
        # 4c. Cap nhat trang thai den tín hieu (gia lap DO lien tuc)
        # ----------------------------------------------------------------
        detector_vio.update_light(fake_red)        # ----------------------------------------------------------------
        # 4d. Nap frame vao buffer bang chung
        # ----------------------------------------------------------------
        extractor.push_frame(frame_idx, frame)

        # ----------------------------------------------------------------
        # 4e. Kiem tra vi pham cho tung xe
        # ----------------------------------------------------------------
        for track in active_tracks:
            event = detector_vio.check(track, frame_idx)
            if event is not None:
                violated_ids.add(track.track_id)
                all_violations.append(event)
                logger.warning(
                    f"[frame {frame_idx:5d}] 🚨 VI PHAM: Track #{track.track_id} "
                    f"({track.class_name}) | ID={event.violation_id}"
                )
                # Luu bang chung ngay lap tuc
                try:
                    ev_path = extractor.save(event, crossing_frame=frame)
                    logger.info(f"           Bang chung luu tai: {ev_path}")
                except Exception as e:
                    logger.error(f"           Loi luu bang chung: {e}")

        # ----------------------------------------------------------------
        # 4f. Ve overlay len frame
        # ----------------------------------------------------------------
        annotated = draw_virtual_lines(frame.copy())
        for track in active_tracks:
            has_crossed = track.track_id in detector_vio._crossed_stop_info
            is_vio      = track.track_id in violated_ids
            annotated   = draw_track(annotated, track, is_vio, has_crossed)

        # Tinh FPS
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0
        if frame_idx % fps_window == 0 and frame_idx > 0:
            fps_display = fps_window / (time.perf_counter() - fps_timer)
            fps_timer = time.perf_counter()

        annotated = draw_hud(
            annotated, frame_idx, fps_display,
            len(active_tracks), len(all_violations),
        )

        # Ghi frame vao video xuat
        if out_writer is not None:
            out_writer.write(annotated)

        # In tien do moi 60 frame
        if frame_idx % 60 == 0:
            logger.info(
                f"  Frame {frame_idx:4d}/{total_frames} | "
                f"FPS={fps_display:.1f} | Tracks={len(active_tracks)} | "
                f"Vi pham={len(all_violations)}"
            )

        frame_idx += 1

    # ------------------------------------------------------------------
    # 5. Ket thuc - giai phong tai nguyen
    # ------------------------------------------------------------------
    cap.release()
    if out_writer is not None:
        out_writer.release()
        logger.info(f"Video ket qua da xuat: {output_path}")

    # ------------------------------------------------------------------
    # 6. Tong ket ket qua
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TONG KET KIEM THU VI PHAM (GD6)")
    print("=" * 60)
    print(f"Tong so frame da xu ly : {frame_idx}")
    print(f"Tong so ID theo doi    : {len(track_states)}")
    print(f"Tong so VI PHAM phat hien: {len(all_violations)}")
    print("-" * 60)

    if all_violations:
        print("Chi tiet cac su kien vi pham:")
        for ev in all_violations:
            print(
                f"  ID={ev.violation_id} | Track #{ev.track_id} ({ev.vehicle_class}) | "
                f"frame_stop={ev.stop_line_frame} | frame_confirm={ev.confirm_line_frame}"
            )
            if ev.evidence_path:
                print(f"    -> Bang chung: {ev.evidence_path}")
    else:
        print("  (Khong phat hien vi pham nao trong video nay)")

    print("=" * 60)
    print("[GD6] Kiem thu hoan tat.")


# ===========================================================================
# Ham phu: Tao video test tong hop (synthetic) neu khong co video thuc
# ===========================================================================

def _create_synthetic_video(out_path: str, duration_sec: int = 15, fps: int = 25) -> str:
    """
    Tao video test tong hop voi xe ao di chuyen theo chieu doc (y tang dan).
    Mot so xe se cat ca 2 vach (vi pham), mot so se dung lai sau vach 1.

    Returns:
        str: Duong dan file video vua tao.
    """
    logger.info("Dang tao video test tong hop (synthetic)...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    width, height = 1000, 600
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total = duration_sec * fps

    # 3 xe: xe 1 va xe 3 vi pham, xe 2 dung truoc vach xac nhan
    class FakeCar:
        def __init__(self, x_center, start_y, end_y, color):
            self.x = x_center
            self.y = float(start_y)
            self.end_y = float(end_y)
            self.color = color
            self.w, self.h = 70, 40

    cars = [
        FakeCar(250, 300, 490, (255, 100, 100)),   # Xe 1: vuot ca 2 vach (VI PHAM)
        FakeCar(500, 300, 420, (100, 255, 100)),   # Xe 2: dung o vach xac nhan (OK)
        FakeCar(750, 300, 480, (100, 100, 255)),   # Xe 3: vuot ca 2 vach (VI PHAM)
    ]

    for fi in range(total):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)   # nen xam

        # Ve vach ke duong
        for yl in range(0, width, 60):
            cv2.line(canvas, (yl, 0), (yl, height), (60, 60, 60), 1)

        # Ve 2 vach ao
        sl_y, cl_y = STOP_LINE_COORDS[0][1], CONFIRM_LINE_COORDS[0][1]
        cv2.line(canvas, (0, sl_y), (width, sl_y), (0, 0, 200), 2)
        cv2.line(canvas, (0, cl_y), (width, cl_y), (0, 200, 200), 2)

        # Di chuyen cac xe
        t = fi / max(total - 1, 1)
        for car in cars:
            car.y = car.y + (car.end_y - car.y) * (2.0 / total)
            car.y = min(car.y, car.end_y)
            x1 = int(car.x - car.w // 2)
            y1 = int(car.y - car.h // 2)
            x2 = int(car.x + car.w // 2)
            y2 = int(car.y + car.h // 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), car.color, -1)

        # Nhan frame
        cv2.putText(canvas, f"Frame {fi} - Synthetic Test", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(canvas)

    writer.release()
    logger.info(f"Da tao video test: {out_path} ({total} frames, {duration_sec}s @ {fps}fps)")
    return out_path


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kiem thu Logic Phat hien Vi pham Vuot Den Do (GD6)"
    )
    parser.add_argument(
        "--video", default=DEFAULT_VIDEO,
        help=f"Duong dan video dau vao (mac dinh: {DEFAULT_VIDEO})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Duong dan model YOLO .pt (mac dinh: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Duong dan video ket qua (mac dinh: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--evidence", default=DEFAULT_EVIDENCE,
        help=f"Thu muc luu bang chung (mac dinh: {DEFAULT_EVIDENCE})"
    )
    args = parser.parse_args()

    run_test(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        evidence_dir=args.evidence,
    )


if __name__ == "__main__":
    main()
