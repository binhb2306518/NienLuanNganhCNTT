"""
scripts/benchmark_tracking.py - So sanh thuc nghiem ByteTrack vs BoT-SORT
========================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ
Giai đoạn 5: Thực nghiệm, Huấn luyện và Tối ưu hóa mô hình

Mục đích:
    - Đo đạc định lượng đối chứng giữa hai thuật toán MOT hàng đầu: ByteTrack và BoT-SORT.
    - So sánh trên cùng một video đầu vào chuẩn 30 giây (900 khung hình).
    - Các tiêu chí khoa học: FPS, độ trễ ms/frame, số lượng ID sinh ra, ước tính ID Switches, và mức VRAM.
    - Cung cấp số liệu thực nghiệm đưa trực tiếp vào Báo cáo Niên luận.
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BenchmarkTracking")


def evaluate_tracker_on_video(
    tracker_name: str,
    video_path: str,
    model_path: str = "models/pretrained/yolov8s.pt",
    device: str = "cuda:0",
    max_frames: int = 450,
) -> Dict[str, float]:
    """
    Chạy suy luận và theo dõi trên video, thu thập số liệu hiệu năng và độ ổn định ID.
    """
    logger.info(f"--- BẮT ĐẦU THỬ NGHIỆM TRACKER: {tracker_name.upper()} ---")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

    model = YOLO(model_path)
    if "cuda" in device and torch.cuda.is_available():
        model.to(device)
        try:
            model.model.half()
        except Exception:
            pass

    tracker_config = f"{tracker_name}.yaml"
    latencies: List[float] = []
    observed_ids = set()
    prev_centroids: Dict[int, Tuple[int, int]] = {}
    id_switches = 0
    frame_idx = 0

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device=device, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    while cap.isOpened() and frame_idx < limit_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        results = model.track(
            source=frame,
            conf=0.40,
            iou=0.45,
            device=device,
            persist=True,
            tracker=tracker_config,
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        current_centroids: Dict[int, Tuple[int, int]] = {}
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.int().cpu().tolist()

            for tid, xyxy in zip(ids, xyxys):
                observed_ids.add(tid)
                cx = (xyxy[0] + xyxy[2]) // 2
                cy = (xyxy[1] + xyxy[3]) // 2
                current_centroids[tid] = (cx, cy)

                # Heuristic ước lượng ID Switch: Nếu một xe ở rất gần vị trí xe frame trước nhưng đổi ID mới
                if tid not in prev_centroids:
                    for prev_id, (pcx, pcy) in prev_centroids.items():
                        if prev_id not in ids:
                            dist = np.hypot(cx - pcx, cy - pcy)
                            if dist < 30:  # Khoảng cách quá gần nhau trong 1 frame
                                id_switches += 1

        prev_centroids = current_centroids

    cap.release()
    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / max(1e-4, total_time)
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    logger.info(f"Hoàn thành {tracker_name}: {avg_fps:.1f} FPS, {len(observed_ids)} IDs, ~{id_switches} switches.")

    return {
        "tracker": tracker_name,
        "total_frames": frame_idx,
        "avg_fps": avg_fps,
        "avg_latency_ms": avg_latency,
        "total_unique_ids": len(observed_ids),
        "id_switches": id_switches,
        "vram_mb": vram_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh ByteTrack vs BoT-SORT trên GPU RTX 3050")
    parser.add_argument("--video", type=str, default="dataset/test_30s_benchmark.mp4", help="Video thử nghiệm")
    parser.add_argument("--model", type=str, default="models/pretrained/yolov8s.pt", help="Trọng số YOLO")
    parser.add_argument("--frames", type=int, default=450, help="Số frame chạy test mỗi thuật toán")
    parser.add_argument("--device", type=str, default="cuda:0", help="Thiết bị tính toán")
    parser.add_argument("--output-csv", type=str, default="reports/tracking_comparison.csv", help="File lưu CSV")

    args = parser.parse_args()

    trackers = ["bytetrack", "botsort"]
    results = []

    for trk in trackers:
        res = evaluate_tracker_on_video(
            tracker_name=trk,
            video_path=args.video,
            model_path=args.model,
            device=args.device,
            max_frames=args.frames,
        )
        results.append(res)

    # Ghi CSV
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Thuật toán", "Frames", "FPS", "Độ trễ (ms)", "Số ID sinh ra", "Ước tính ID Switches", "VRAM (MB)"])
        for r in results:
            writer.writerow([
                r["tracker"],
                r["total_frames"],
                f"{r['avg_fps']:.2f}",
                f"{r['avg_latency_ms']:.2f}",
                r["total_unique_ids"],
                r["id_switches"],
                f"{r['vram_mb']:.1f}",
            ])

    print("\n" + "=" * 80)
    print("         BẢNG SO SÁNH HIỆU NĂNG THEO DÕI ĐA MỤC TIÊU (MOT)")
    print("=" * 80)
    print(f"{'Thuật toán':<15} | {'FPS':<10} | {'Độ trễ (ms)':<14} | {'Tổng ID':<10} | {'ID Switches':<12} | {'VRAM'}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['tracker']:<15} | "
            f"{r['avg_fps']:<10.2f} | "
            f"{r['avg_latency_ms']:<14.2f} | "
            f"{r['total_unique_ids']:<10d} | "
            f"{r['id_switches']:<12d} | "
            f"{r['vram_mb']:.1f} MB"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

