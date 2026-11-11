"""
evaluate.py - Danh gia hieu nang mo hinh YOLO tren tap Test doc lap
====================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ
Giai đoạn 5: Thực nghiệm, Huấn luyện và Tối ưu hóa mô hình

Mục đích:
    - Đánh giá định lượng mô hình YOLO trên tập Test độc lập (tránh data leakage).
    - Đo lường chuẩn khoa học: mAP@0.5, mAP@0.5:0.95, Precision, Recall cho từng lớp.
    - Đo độ trễ suy luận chi tiết (Preprocess, Inference, Postprocess) trên GPU RTX 3050.
    - So sánh trực diện kết quả giữa các mô hình ứng viên (YOLOv8s vs YOLO11s).

Cách dùng:
    # Đánh giá YOLOv8s trên tập test:
    python evaluate.py --model models/pretrained/yolov8s.pt --split test

    # Đánh giá YOLO11s hoặc checkpoint đã fine-tune:
    python evaluate.py --model models/checkpoints/best.pt --split test
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateYOLO")


def evaluate_model(
    model_path: str = "models/pretrained/yolov8s.pt",
    data_yaml: str = "dataset/data.yaml",
    split: str = "test",
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.60,
) -> Dict[str, Any]:
    """
    Thực hiện đánh giá mô hình YOLO trên tập dữ liệu được cấu hình.
    """
    import torch
    from ultralytics import YOLO

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    logger.info("=" * 70)
    logger.info("  ĐÁNH GIÁ MÔ HÌNH PHÁT HIỆN ĐỐI TƯỢNG (OBJECT DETECTION)")
    logger.info(f"  • Trọng số mô hình:   {model_path}")
    logger.info(f"  • Cấu hình Dataset:   {data_yaml}")
    logger.info(f"  • Tập dữ liệu đánh giá: {split}")
    logger.info(f"  • Thiết bị phần cứng: {device} ({device_name})")
    logger.info(f"  • Confidence: {conf_thresh} | IoU: {iou_thresh} | ImgSz: {img_size}")
    logger.info("=" * 70)

    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {data_path}")

    # Nạp mô hình
    logger.info(f"Đang nạp mô hình từ {model_path}...")
    model = YOLO(model_path)

    # Chạy validation/test
    logger.info(f"Bắt đầu quá trình đánh giá trên tập {split.upper()}...")
    try:
        metrics = model.val(
            data=str(data_path),
            split=split,
            batch=batch_size,
            imgsz=img_size,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device if cuda_available else "cpu",
            half=cuda_available,
            plots=True,
            save_json=False,
            verbose=True,
        )

        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        # Đo tốc độ
        speed_dict = metrics.speed
        inference_ms = speed_dict.get("inference", 0.0)
        total_ms = sum(speed_dict.values())
        fps_val = 1000.0 / total_ms if total_ms > 0 else 0.0

        print("\n" + "=" * 70)
        print("               KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
        print("=" * 70)
        print(f"  • mAP@0.5:             {map50 * 100:6.2f} %")
        print(f"  • mAP@0.5:0.95:        {map50_95 * 100:6.2f} %")
        print(f"  • Mean Precision (P):  {precision * 100:6.2f} %")
        print(f"  • Mean Recall (R):     {recall * 100:6.2f} %")
        print(f"  • Độ trễ suy luận:    {inference_ms:6.2f} ms / ảnh")
        print(f"  • Tốc độ tính toán:    {fps_val:6.2f} FPS ({device_name})")
        print("=" * 70 + "\n")

        return {
            "model": model_path,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "inference_ms": inference_ms,
            "fps": fps_val,
        }

    except Exception as e:
        logger.warning(f"Không thể thực thi model.val trực tiếp (có thể do tập test trống): {e}")
        logger.info("Chuyển sang chế độ kiểm tra suy luận mẫu (Sample Inference Benchmark)...")

        # Kiểm tra suy luận trên 100 frames giả lập
        import numpy as np
        import time
        dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            model.predict(dummy, device=device if cuda_available else "cpu", verbose=False)
        
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict(dummy, device=device if cuda_available else "cpu", verbose=False)
            if cuda_available:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        avg_latency = float(np.mean(times))
        fps_sim = 1000.0 / avg_latency

        print("\n" + "=" * 70)
        print("           KẾT QUẢ ĐO LƯỜNG TỐC ĐỘ SUY LUẬN MÔ HÌNH")
        print("=" * 70)
        print(f"  • Mô hình kiểm tra:    {model_path}")
        print(f"  • Thiết bị phần cứng:  {device_name}")
        print(f"  • Độ trễ trung bình:   {avg_latency:6.2f} ms / ảnh")
        print(f"  • Tốc độ suy luận:     {fps_sim:6.2f} FPS")
        print("=" * 70 + "\n")

        return {
            "model": model_path,
            "inference_ms": avg_latency,
            "fps": fps_sim,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình YOLO trên tập Test độc lập",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="models/pretrained/yolov8s.pt", help="Đường dẫn file mô hình .pt")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Tập dữ liệu đánh giá")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Chỉ số thiết bị GPU hoặc 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy conf")
    parser.add_argument("--iou", type=float, default=0.60, help="Ngưỡng IoU")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )


if __name__ == "__main__":
    main()

