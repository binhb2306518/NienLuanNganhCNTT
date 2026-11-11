"""
run_baseline.py - Trien khai Baseline Pipeline (YOLOv8s + ByteTrack)
=====================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ
Giai đoạn 4: Xây dựng Baseline Pipeline tối thiểu

Mục đích:
    - Kết nối luồng hoàn chỉnh từ Video đầu vào -> VideoLoader -> Detector (YOLOv8s FP16)
      -> Tracker (ByteTrack) -> Overlay Visualizer -> VideoWriter (annotated_baseline.mp4).
    - Đo đạc chính xác hiệu năng FPS, độ trễ từng frame (ms), và mức tiêu thụ VRAM trên GPU RTX 3050.
    - Làm mốc đối chứng (Baseline benchmark) khoa học cho các cải tiến tiếp theo.

Cách dùng:
    python run_baseline.py
    python run_baseline.py --source my_video.mp4 --output annotated_baseline.mp4
    python run_baseline.py --source 0  (Webcam)
    python run_baseline.py --max-frames 900  (Chạy chuẩn 30s ở 30fps)
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Thiết lập đường dẫn gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BaselinePipeline")


# ─── BẢNG MÀU TỐI ƯU TRỰC QUAN THEO ID VÀ LỚP ─────────────────────────────────
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (56, 178, 103),        # Xanh lá cây
    "motorcycle": (255, 140, 0),   # Cam rực rỡ
    "bus": (230, 81, 0),          # Xanh dương đậm
    "truck": (30, 144, 255),       # Đỏ cam
    "traffic_light": (0, 215, 255),# Vàng rực rỡ
}

def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Sinh màu cố định duy nhất dựa trên ID của track để dễ quan sát."""
    np.random.seed(track_id * 37)
    color = np.random.randint(60, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


# ─── BỘ TẠO VIDEO BENCHMARK 30S TỰ ĐỘNG (KHI CHƯA CÓ VIDEO THỰC TẾ) ───────────
def generate_synthetic_benchmark_video(
    output_path: str = "dataset/test_30s_benchmark.mp4",
    duration_sec: int = 30,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> str:
    """
    Tự động sinh video giao thông mô phỏng 30 giây (900 khung hình) chuẩn 720p.
    Bao gồm các phương tiện di chuyển qua lại, vạch dừng và đèn tín hiệu,
    giúp nghiệm thu hiệu năng baseline ngay lập tức mà không cần chuẩn bị trước video.
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000:
        logger.info(f"Video benchmark đã tồn tại: {output_path}")
        return output_path

    logger.info(f"Đang sinh video benchmark mô phỏng {duration_sec}s ({duration_sec * fps} frames)...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Trạng thái các phương tiện giả lập
    vehicles = [
        {"x": 200, "y": 100, "vx": 0, "vy": 4.5, "w": 70, "h": 120, "color": (200, 200, 200), "name": "car"},
        {"x": 420, "y": 50,  "vx": 0, "vy": 3.8, "w": 65, "h": 110, "color": (180, 100, 50), "name": "car"},
        {"x": 650, "y": 20,  "vx": 0, "vy": 5.2, "w": 40, "h": 70,  "color": (50, 150, 255), "name": "motorcycle"},
        {"x": 800, "y": 80,  "vx": 0, "vy": 4.0, "w": 38, "h": 68,  "color": (80, 200, 80), "name": "motorcycle"},
        {"x": 1000,"y": 10,  "vx": 0, "vy": 3.0, "w": 90, "h": 180, "color": (50, 50, 200), "name": "bus"},
    ]

    for f in range(total_frames):
        # Mặt đường màu xám
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        # Làn đường
        cv2.line(frame, (350, 0), (350, height), (80, 80, 80), 2)
        cv2.line(frame, (580, 0), (580, height), (80, 80, 80), 2)
        cv2.line(frame, (750, 0), (750, height), (80, 80, 80), 2)
        cv2.line(frame, (950, 0), (950, height), (80, 80, 80), 2)

        # Vạch dừng màu trắng (Stop Line)
        cv2.line(frame, (100, 480), (1180, 480), (255, 255, 255), 6)
        cv2.putText(frame, "STOP LINE", (120, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Đèn tín hiệu giao thông (góc trên bên phải)
        cv2.rectangle(frame, (1150, 40), (1220, 220), (20, 20, 20), -1)
        cv2.rectangle(frame, (1150, 40), (1220, 220), (150, 150, 150), 2)
        
        # Chu kỳ đèn: 15s Đỏ, 12s Xanh, 3s Vàng
        cycle_sec = (f / fps) % 30
        if cycle_sec < 15:
            cv2.circle(frame, (1185, 80), 22, (0, 0, 255), -1)    # Đỏ
            cv2.circle(frame, (1185, 130), 18, (30, 30, 30), -1)  # Vàng tắt
            cv2.circle(frame, (1185, 180), 18, (30, 30, 30), -1)  # Xanh tắt
        elif cycle_sec < 27:
            cv2.circle(frame, (1185, 80), 18, (30, 30, 30), -1)   # Đỏ tắt
            cv2.circle(frame, (1185, 130), 18, (30, 30, 30), -1)  # Vàng tắt
            cv2.circle(frame, (1185, 180), 22, (0, 255, 0), -1)   # Xanh bật
        else:
            cv2.circle(frame, (1185, 80), 18, (30, 30, 30), -1)   # Đỏ tắt
            cv2.circle(frame, (1185, 130), 22, (0, 255, 255), -1) # Vàng bật
            cv2.circle(frame, (1185, 180), 18, (30, 30, 30), -1)  # Xanh tắt

        # Vẽ và cập nhật xe
        for v in vehicles:
            vx, vy = int(v["x"]), int(v["y"])
            vw, vh = int(v["w"]), int(v["h"])
            cv2.rectangle(frame, (vx - vw // 2, vy - vh // 2), (vx + vw // 2, vy + vh // 2), v["color"], -1)
            cv2.rectangle(frame, (vx - vw // 2, vy - vh // 2), (vx + vw // 2, vy + vh // 2), (255, 255, 255), 2)
            cv2.putText(frame, v["name"], (vx - vw // 2, vy - vh // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Di chuyển xe xuống dưới
            v["y"] += v["vy"]
            if v["y"] > height + 100:
                v["y"] = -50 - np.random.randint(0, 100)

        writer.write(frame)

    writer.release()
    logger.info(f"Đã sinh video benchmark 30s thành công tại: {output_path}")
    return output_path


# ─── LỚP VẼ OVERLAY GIAO DIỆN CHUYÊN NGHIỆP TỐI ƯU TỐC ĐỘ ──────────────────────
class Visualizer:
    """
    Module vẽ bounding box, ID, nhãn, đuôi quỹ đạo và HUD thống kê hiệu năng.
    Tối ưu hóa các thao tác OpenCV trực tiếp trên NumPy buffer để đạt FPS cao nhất.
    """

    def __init__(self, history_len: int = 30) -> None:
        self.history_len = history_len
        self.track_histories: Dict[int, deque] = {}

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
    ) -> np.ndarray:
        """
        Vẽ Bounding Box, Track ID, nhãn lớp, và đuôi quỹ đạo (Trajectory).
        """
        for t in tracks:
            tid = t["id"]
            box = t["box"]
            cls_name = t["class_name"]
            conf = t["conf"]
            x1, y1, x2, y2 = box

            # Cập nhật lịch sử tâm đáy
            cx = (x1 + x2) // 2
            bottom_center = (cx, y2)
            if tid not in self.track_histories:
                self.track_histories[tid] = deque(maxlen=self.history_len)
            self.track_histories[tid].append(bottom_center)

            # Chọn màu theo lớp hoặc theo ID
            color = CLASS_COLORS.get(cls_name, get_track_color(tid))

            # 1. Vẽ đuôi quỹ đạo di chuyển
            pts = list(self.track_histories[tid])
            for i in range(1, len(pts)):
                thickness = int(np.sqrt(float(i + 1)) * 1.5)
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)

            # 2. Vẽ Bounding Box với góc nhấn hiện đại
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, bottom_center, 4, (0, 0, 255), -1)

            # 3. Vẽ nhãn có nền (Label badge)
            label = f"ID:{tid} {cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_y1 = max(0, y1 - lh - 8)
            label_y2 = y1
            cv2.rectangle(
                frame,
                (x1, label_y1),
                (x1 + lw + 6, label_y2),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 3, label_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        frame_idx: int,
        total_frames: int,
        active_tracks_count: int,
        vram_mb: float = 0.0,
        device_name: str = "GPU",
    ) -> np.ndarray:
        """
        Vẽ bảng điều khiển HUD (Heads-Up Display) hiển thị số liệu hiệu năng khoa học.
        """
        overlay = frame.copy()
        hud_w, hud_h = 360, 160
        cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (80, 90, 100), 1)

        # Tiêu đề HUD
        cv2.putText(
            frame,
            "BASELINE PIPELINE: YOLOv8s + ByteTrack",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        # Dòng 1: FPS & Độ trễ
        fps_color = (0, 255, 127) if fps >= 25.0 else (0, 165, 255)
        cv2.putText(
            frame,
            f"FPS: {fps:5.1f}  |  Latency: {latency_ms:5.1f} ms",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            fps_color,
            2,
            cv2.LINE_AA,
        )

        # Dòng 2: Thiết bị & VRAM
        vram_text = f"Device: {device_name} (VRAM: {vram_mb:.0f} MB)" if vram_mb > 0 else f"Device: {device_name}"
        cv2.putText(
            frame,
            vram_text,
            (25, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Dòng 3: Số track đang hoạt động
        cv2.putText(
            frame,
            f"Active Tracks: {active_tracks_count} | Precision: FP16",
            (25, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Dòng 4: Tiến độ khung hình
        progress_pct = (frame_idx / max(1, total_frames)) * 100 if total_frames > 0 else 0
        cv2.putText(
            frame,
            f"Frame: {frame_idx}/{total_frames} ({progress_pct:.1f}%)",
            (25, 148),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        return frame


# ─── HÀM LẤY MỨC TIÊU THỤ VRAM GPU ─────────────────────────────────────────────
def get_gpu_vram_mb() -> float:
    """Trả về dung lượng VRAM đang được PyTorch cấp phát (MB)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ─── HÀM THỰC THI PIPELINE CHÍNH ──────────────────────────────────────────────
def run_baseline(
    source: str,
    output_path: str = "annotated_baseline.mp4",
    model_path: str = "models/pretrained/yolov8s.pt",
    tracker_type: str = "bytetrack",
    conf_thresh: float = 0.40,
    iou_thresh: float = 0.45,
    img_size: int = 640,
    device: str = "cuda:0",
    use_fp16: bool = True,
    max_frames: Optional[int] = None,
    show_window: bool = False,
) -> Dict[str, float]:
    """
    Thực thi Baseline Pipeline: YOLOv8s + ByteTrack trên luồng video.

    Returns:
        Dict chứa các chỉ số hiệu năng tổng kết (total_frames, avg_fps, avg_latency_ms, peak_vram_mb).
    """
    import torch
    from ultralytics import YOLO

    # 1. Kiểm tra phần cứng CUDA
    actual_device = device
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA không khả dụng! Fallback về CPU (FP32).")
        actual_device = "cpu"
        use_fp16 = False

    device_name = "CPU"
    if "cuda" in actual_device:
        device_name = torch.cuda.get_device_name(0)

    logger.info("=" * 70)
    logger.info("  BẮT ĐẦU BASELINE PIPELINE (YOLOv8s + ByteTrack)")
    logger.info(f"  • Source Video:    {source}")
    logger.info(f"  • Output Video:    {output_path}")
    logger.info(f"  • Model Path:      {model_path}")
    logger.info(f"  • Tracker:         {tracker_type}")
    logger.info(f"  • Hardware Device: {actual_device} ({device_name})")
    logger.info(f"  • Precision:       {'FP16 (Tensor Cores)' if use_fp16 else 'FP32'}")
    logger.info(f"  • Confidence:      {conf_thresh} | IoU: {iou_thresh} | ImgSz: {img_size}")
    logger.info("=" * 70)

    # 2. Nạp mô hình YOLOv8s
    if not os.path.exists(model_path):
        logger.info(f"Model chưa có tại {model_path}, Ultralytics sẽ tự động nạp từ 'yolov8s.pt'...")
        model = YOLO("yolov8s.pt")
    else:
        logger.info(f"Đang nạp trọng số mô hình từ: {model_path}...")
        model = YOLO(model_path)

    # Chuyển mô hình sang thiết bị và tối ưu bán chính xác FP16
    try:
        model.to(actual_device)
        if use_fp16 and "cuda" in actual_device:
            model.model.half()
            logger.info("Đã chuyển đổi trọng số mô hình sang FP16 Half-Precision (Tensor Cores).")
    except Exception as e:
        logger.warning(f"Lỗi khi chuyển đổi FP16: {e}. Sử dụng FP32 tiêu chuẩn.")
        use_fp16 = False

    # 3. Mở luồng video đầu vào
    # Hỗ trợ cả webcam số nguyên (ví dụ: '0')
    cap_src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở nguồn video: {source}")

    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

    logger.info(f"Video input: {in_width}x{in_height} @ {in_fps:.1f} FPS | Tổng số frames: {total_frames}")

    # 4. Khởi tạo VideoWriter xuất video đã vẽ overlay
    os.makedirs(Path(output_path).parent or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, in_fps, (in_width, in_height))

    # 5. Khởi tạo Visualizer
    visualizer = Visualizer(history_len=30)

    # 6. Warm-up GPU trước khi bấm giờ chính thức
    if "cuda" in actual_device:
        logger.info("Đang thực hiện GPU Warm-up (nạp CUDA kernels)...")
        dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        model.predict(dummy, device=actual_device, verbose=False)
        torch.cuda.synchronize()
        logger.info("Hoàn tất warm-up GPU.")

    # 7. Vòng lặp xử lý chính (Processing Loop)
    frame_idx = 0
    latencies: List[float] = []
    fps_smooth = in_fps
    peak_vram = 0.0

    # Lọc các lớp xe và đèn giao thông nếu dùng pretrained COCO
    # COCO classes: 2:car, 3:motorcycle, 5:bus, 7:truck, 9:traffic light
    coco_vehicle_classes = [2, 3, 5, 7, 9]
    # Nếu mô hình tùy chỉnh có ít lớp (<= 5 lớp) thì sử dụng toàn bộ lớp
    filter_classes = None
    if hasattr(model, "names") and len(model.names) > 20:
        filter_classes = coco_vehicle_classes
        logger.info("Mô hình phát hiện COCO 80 lớp: tự động lọc các lớp xe và đèn [2, 3, 5, 7, 9].")

    tracker_config = f"{tracker_type}.yaml"
    start_total_time = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            t0 = time.perf_counter()

            # --- A. BƯỚC SUY LUẬN + TRACKING TỐI ƯU TRÊN CUDA ---
            # Ultralytics model.track kết hợp YOLOv8s + ByteTrack song song trên GPU
            results = model.track(
                source=frame,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=img_size,
                device=actual_device,
                classes=filter_classes,
                persist=True,
                tracker=tracker_config,
                verbose=False,
            )

            # Đồng bộ CUDA để đo độ trễ phần cứng chính xác
            if "cuda" in actual_device:
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            frame_latency_ms = (t1 - t0) * 1000.0
            latencies.append(frame_latency_ms)

            # Làm mượt chỉ số FPS hiển thị (EMA alpha=0.15)
            instant_fps = 1000.0 / max(1e-3, frame_latency_ms)
            fps_smooth = 0.85 * fps_smooth + 0.15 * instant_fps

            # --- B. TRÍCH XUẤT CÁC ĐỐI TƯỢNG TRACKING ---
            tracks_info: List[Dict] = []
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.int().cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.float().cpu().tolist()
                xyxys = boxes.xyxy.int().cpu().tolist()

                for tid, cid, conf_val, xyxy in zip(ids, clss, confs, xyxys):
                    cname = model.names.get(cid, str(cid)) if hasattr(model, "names") else str(cid)
                    tracks_info.append({
                        "id": tid,
                        "class_id": cid,
                        "class_name": cname,
                        "conf": conf_val,
                        "box": xyxy,
                    })

            # Đo dung lượng VRAM GPU
            current_vram = get_gpu_vram_mb()
            if current_vram > peak_vram:
                peak_vram = current_vram

            # --- C. VẼ OVERLAY VÀ BẢNG ĐIỀU KHIỂN HUD ---
            annotated_frame = visualizer.draw_tracks(frame, tracks_info)
            annotated_frame = visualizer.draw_hud(
                frame=annotated_frame,
                fps=fps_smooth,
                latency_ms=frame_latency_ms,
                frame_idx=frame_idx,
                total_frames=total_frames,
                active_tracks_count=len(tracks_info),
                vram_mb=current_vram,
                device_name=device_name,
            )

            # --- D. GHI VIDEO RA TẬP TIN ---
            writer.write(annotated_frame)

            # Hiển thị trực tiếp nếu yêu cầu
            if show_window:
                cv2.imshow("Baseline Pipeline - YOLOv8s + ByteTrack", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Người dùng bấm phím 'q' để dừng quá trình.")
                    break

            # In log tiến độ định kỳ mỗi 60 frame
            if frame_idx % 60 == 0 or frame_idx == total_frames:
                logger.info(
                    f"Frame [{frame_idx:4d}/{total_frames:4d}] "
                    f"FPS: {fps_smooth:5.1f} | "
                    f"Latency: {frame_latency_ms:5.1f} ms | "
                    f"Tracks: {len(tracks_info):2d} | "
                    f"VRAM: {current_vram:4.0f} MB"
                )

    finally:
        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    total_processing_time = time.perf_counter() - start_total_time
    avg_fps = frame_idx / max(1e-4, total_processing_time)
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    # 8. BÁO CÁO THỐNG KÊ KHOA HỌC
    print("\n" + "=" * 70)
    print("        KẾT QUẢ NGHIỆM THU HIỆU NĂNG BASELINE PIPELINE")
    print("=" * 70)
    print(f"  • Tổng số khung hình đã xử lý:  {frame_idx} frames")
    print(f"  • Tổng thời gian thực thi:       {total_processing_time:.2f} giây")
    print(f"  • TỐC ĐỘ FPS TRUNG BÌNH:        {avg_fps:.2f} FPS   {'[ĐẠT CHỈ TIÊU >= 25 FPS ✓]' if avg_fps >= 25 else '[CHƯA ĐẠT]'}")
    print(f"  • Độ trễ trung bình / frame:     {avg_latency:.2f} ms")
    print(f"  • Độ trễ phân vị P95 / frame:    {p95_latency:.2f} ms")
    print(f"  • Mức tiêu thụ VRAM đỉnh:        {peak_vram:.1f} MB (trên 6144 MB RTX 3050)")
    print(f"  • Tập tin video kết quả:         {Path(output_path).resolve()}")
    print("=" * 70 + "\n")

    return {
        "total_frames": float(frame_idx),
        "total_time_sec": float(total_processing_time),
        "avg_fps": float(avg_fps),
        "avg_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
        "peak_vram_mb": float(peak_vram),
    }


# ─── ENTRY POINT CLI ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy Baseline Pipeline YOLOv8s + ByteTrack trên GPU RTX 3050",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Đường dẫn file video MP4, webcam ID (0), hoặc URL RTSP. Để trống sẽ tự sinh video benchmark 30s.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="annotated_baseline.mp4",
        help="Đường dẫn file video kết quả đầu ra (MP4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pretrained/yolov8s.pt",
        help="Đường dẫn mô hình YOLOv8s pretrained.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Thuật toán MOT tracker.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Ngưỡng tin cậy phát hiện (Confidence threshold).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Ngưỡng IoU Non-Maximum Suppression.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào mô hình YOLO.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Thiết bị tính toán: 'cuda:0' hoặc 'cpu'.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Tắt chế độ FP16 nửa chính xác (dùng FP32 chuẩn).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=900,
        help="Giới hạn số khung hình cần xử lý (mặc định 900 frame = 30 giây ở 30 FPS).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị cửa sổ xem trực tiếp kết quả (chỉ dùng khi có màn hình).",
    )

    args = parser.parse_args()

    # Xử lý nguồn video đầu vào: Nếu người dùng không chỉ định hoặc file không tồn tại, tự sinh video test 30s
    video_source = args.source
    if not video_source or (not video_source.isdigit() and not os.path.exists(video_source)):
        benchmark_path = "dataset/test_30s_benchmark.mp4"
        logger.info(f"Không có nguồn video chỉ định hoặc không tìm thấy '{video_source}'.")
        logger.info(f"Khởi động chế độ Benchmark: Tự động sinh video giao thông 30 giây tại: {benchmark_path}")
        video_source = generate_synthetic_benchmark_video(benchmark_path, duration_sec=30, fps=30)

    # Chạy pipeline
    results = run_baseline(
        source=video_source,
        output_path=args.output,
        model_path=args.model,
        tracker_type=args.tracker,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=args.imgsz,
        device=args.device,
        use_fp16=not args.no_fp16,
        max_frames=args.max_frames,
        show_window=args.show,
    )

    # Kiểm tra tiêu chí nghiệm thu
    if results["avg_fps"] >= 25.0:
        logger.info(f"✓ CHÚC MỪNG: Baseline Pipeline đạt chuẩn nghiệm thu Giai đoạn 4 với {results['avg_fps']:.1f} FPS!")
    else:
        logger.warning(f"⚠ Baseline Pipeline chạy ở mức {results['avg_fps']:.1f} FPS (chưa vượt ngưỡng 25 FPS).")


if __name__ == "__main__":
    main()
