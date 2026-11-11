"""
train.py - Script huan luyen va tinh chinh (Fine-tune) mo hinh YOLO tren tap du lieu Giao thong
=============================================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ
Giai đoạn 5: Thực nghiệm, Huấn luyện và Tối ưu hóa mô hình

Mục đích:
    - Huấn luyện / fine-tune YOLOv11s hoặc YOLOv8s trên tập dữ liệu đặc thù Việt Nam (5 lớp).
    - Hỗ trợ chạy linh hoạt cả Cục bộ (Local RTX 3050 6GB) lẫn Đám mây (Kaggle GPU T4 / Google Colab).
    - Tự động áp dụng cơ chế Automatic Mixed Precision (AMP - FP16) tiết kiệm VRAM và tăng tốc.
    - Tự động lưu checkpoint tốt nhất 'best.pt' phục vụ tích hợp hệ thống.

Cách dùng:
    # Huấn luyện mặc định (YOLOv11s, batch=16, 50 epochs):
    python train.py

    # Huấn luyện YOLOv8s với batch=8 an toàn cho VRAM:
    python train.py --model yolov8s.pt --batch 8 --epochs 50

    # Chạy trên Kaggle / Colab GPU:
    python train.py --model yolo11s.pt --batch 16 --epochs 50 --device 0 --kaggle
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# Thiết lập root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainYOLO")


def train_yolo(
    model_name: str = "yolo11s.pt",
    data_yaml: str = "dataset/data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    workers: int = 4,
    project: str = "experiments",
    experiment_name: str = "traffic_yolo_run",
    patience: int = 15,
    save_period: int = 5,
    is_kaggle: bool = False,
) -> Path:
    """
    Thực thi quy trình fine-tune mô hình YOLO.

    Returns:
        Path tới tệp trọng số tốt nhất 'best.pt'.
    """
    import torch
    from ultralytics import YOLO

    # 1. Kiểm tra tài nguyên tính toán
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    logger.info("=" * 70)
    logger.info("  CẤU HÌNH HUẤN LUYỆN YOLO CHO DỰ ÁN NIÊN LUẬN AI")
    logger.info(f"  • Mô hình cơ sở:     {model_name}")
    logger.info(f"  • Tệp cấu hình data: {data_yaml}")
    logger.info(f"  • Số Epoch:          {epochs} (Early Stopping: {patience})")
    logger.info(f"  • Batch size:        {batch_size}")
    logger.info(f"  • Kích thước ảnh:    {img_size}x{img_size}")
    logger.info(f"  • Thiết bị phần cứng: {device} ({device_name})")
    logger.info(f"  • Môi trường:        {'Kaggle / Colab Cloud' if is_kaggle else 'Local Machine'}")
    logger.info(f"  • Thư mục lưu kết quả: {project}/{experiment_name}")
    logger.info("=" * 70)

    # Đảm bảo đường dẫn tuyệt đối hoặc hợp lệ cho data.yaml
    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp cấu hình dataset: {data_path}. "
            "Hãy kiểm tra lại đường dẫn tệp data.yaml."
        )

    # 2. Khởi tạo mô hình YOLO
    logger.info(f"Đang nạp mô hình: {model_name}...")
    model = YOLO(model_name)

    # 3. Tiến hành huấn luyện với cấu hình tối ưu VRAM
    logger.info("Bắt đầu huấn luyện...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device if cuda_available else "cpu",
        workers=workers if cuda_available else 2,
        project=project,
        name=experiment_name,
        patience=patience,
        save=True,
        save_period=save_period,
        amp=True,              # Automatic Mixed Precision (FP16) - cực kỳ quan trọng cho RTX 3050 & Kaggle T4
        cache=False,           # Tắt cache RAM nếu RAM hệ thống giới hạn
        plots=True,            # Tự động vẽ đồ thị loss curves và confusion matrix
        exist_ok=True,
        verbose=True,
    )

    # 4. Xác định đường dẫn trọng số tốt nhất
    best_weight = Path(project) / experiment_name / "weights" / "best.pt"
    if best_weight.exists():
        logger.info(f"✓ Huấn luyện thành công! Trọng số tốt nhất đã lưu tại: {best_weight.resolve()}")
        # Copy một bản về models/checkpoints/ để tiện sử dụng
        os.makedirs(PROJECT_ROOT / "models" / "checkpoints", exist_ok=True)
        target_pt = PROJECT_ROOT / "models" / "checkpoints" / f"{experiment_name}_best.pt"
        import shutil
        shutil.copy2(best_weight, target_pt)
        logger.info(f"✓ Đã sao lưu trọng số vào: {target_pt}")
    else:
        logger.warning(f"Không tìm thấy file trọng số tại {best_weight}. Vui lòng kiểm tra thư mục {project}/{experiment_name}")

    return best_weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình YOLOv11s / YOLOv8s trên tập dữ liệu giao thông",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Mô hình pretrained cơ sở (yolo11s.pt | yolov8s.pt)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Đường dẫn file cấu hình data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch (16 hoặc 8 nếu VRAM hạn chế)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", type=str, default="0", help="Chỉ số GPU ('0') hoặc 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="Số worker nạp dữ liệu")
    parser.add_argument("--project", type=str, default="experiments", help="Thư mục gốc lưu kết quả thí nghiệm")
    parser.add_argument("--name", type=str, default="yolo_traffic_finetune", help="Tên thí nghiệm con")
    parser.add_argument("--patience", type=int, default=15, help="Số epochs dừng sớm nếu validation loss không giảm")
    parser.add_argument("--kaggle", action="store_true", help="Bật cờ khi chạy trong môi trường Kaggle Notebook")

    args = parser.parse_args()

    train_yolo(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        experiment_name=args.name,
        patience=args.patience,
        is_kaggle=args.kaggle,
    )


if __name__ == "__main__":
    main()

