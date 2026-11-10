"""
organize_dataset.py — Quy trình tổ chức dữ liệu Video-level Split
===================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình — B2306518 — CT466E — ĐH Cần Thơ

MỤC ĐÍCH
---------
Tổ chức frames và nhãn từ nhiều video thô vào cấu trúc YOLO chuẩn,
đảm bảo Video-level Split: KHÔNG bao giờ chia frame ngẫu nhiên
(tránh Data Leakage giữa Train/Val/Test).

QUY TẮC CHIA TẬP (Video-level Split)
--------------------------------------
  - Train : 70% số video
  - Val   : 15% số video
  - Test  : 15% số video → Tuyệt đối không có frame trùng với Train/Val

CÁCH DÙNG
----------
  # Bước 1: Trích frame từ video thô
  python scripts/organize_dataset.py extract --video_dir videos/raw --fps 2

  # Bước 2: Chia tập dữ liệu (sau khi đã gán nhãn)
  python scripts/organize_dataset.py split --ratio 0.70 0.15 0.15 --seed 42

  # Bước 3: Kiểm tra kết quả
  python scripts/data_checker.py

LƯU Ý
------
  - Gán nhãn với Roboflow / LabelImg / CVAT TRƯỚC KHI chạy split.
  - Frame prefix = tên video gốc → đảm bảo truy vết nguồn gốc.
"""

import os
import sys
import shutil
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT        = Path(__file__).resolve().parent.parent
DATASET     = ROOT / "dataset"
VIDEOS_RAW  = ROOT / "videos" / "raw"
SCRATCH_DIR = DATASET / "_frames_scratch"   # Vùng trung gian trước khi split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("OrgDataset")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS   = ["train", "val", "test"]


# ═══════════════════════════════════════════════════════════════════════════════
# Bước 1: Trích frame từ video thô
# ═══════════════════════════════════════════════════════════════════════════════
def extract_frames(video_dir: Path, fps: float = 2.0, max_frames: int = 500):
    """
    Trích frame từ toàn bộ video trong video_dir với tốc độ `fps` frame/giây.
    Frame được lưu vào dataset/_frames_scratch/<video_name>/frame_XXXXX.jpg

    Parameters
    ----------
    video_dir  : thư mục chứa video .mp4/.avi/.mov
    fps        : số frame trích xuất mỗi giây (mặc định 2 fps)
    max_frames : số frame tối đa mỗi video (tránh mất cân bằng)
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python chưa cài. Chạy: pip install opencv-python")
        sys.exit(1)

    video_exts  = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}
    video_files = [p for p in video_dir.iterdir() if p.suffix in video_exts]

    if not video_files:
        logger.warning(f"Không tìm thấy video nào trong: {video_dir}")
        logger.info("→ Đặt video vào thư mục: D:\\NIENLUANNGANH\\videos\\raw\\")
        return

    logger.info(f"Tìm thấy {len(video_files)} video. Bắt đầu trích frame ({fps} fps)...")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    for vid_path in sorted(video_files):
        vid_name  = vid_path.stem
        out_dir   = SCRATCH_DIR / vid_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            logger.warning(f"  Không mở được video: {vid_path.name}")
            continue

        vid_fps   = cap.get(cv2.CAP_PROP_FPS)
        interval  = max(1, int(round(vid_fps / fps)))  # Lấy 1 frame mỗi `interval` frames
        total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"  {vid_path.name}: {total_vid} frames @ {vid_fps:.1f}fps "
            f"→ trích 1/{interval} = ~{total_vid // interval} frames"
        )

        saved = 0
        frame_idx = 0
        while cap.isOpened() and saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out_path = out_dir / f"{vid_name}_frame{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            frame_idx += 1

        cap.release()
        logger.info(f"    → Đã lưu {saved} frames vào {out_dir.relative_to(ROOT)}")

    logger.info(f"\nHoàn tất trích frame → {SCRATCH_DIR}")
    logger.info("Tiếp theo: Gán nhãn với Roboflow/CVAT, rồi chạy 'split'")


# ═══════════════════════════════════════════════════════════════════════════════
# Bước 2: Video-level Split
# ═══════════════════════════════════════════════════════════════════════════════
def video_level_split(
    ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    labeled_dir: Path = None,
):
    """
    Chia dữ liệu theo Video-level Split.

    Phát hiện các "video group" dựa vào tiền tố filename (phần trước '_frame').
    Mỗi video group được gán NGUYÊN VẸEN vào một split (Train, Val, hoặc Test).

    Parameters
    ----------
    ratio      : (train_ratio, val_ratio, test_ratio), tổng = 1.0
    seed       : random seed để tái tạo kết quả
    labeled_dir: thư mục chứa ảnh đã gán nhãn (.jpg + .txt cùng thư mục),
                 mặc định dùng dataset/_frames_scratch/
    """
    assert abs(sum(ratio) - 1.0) < 1e-6, "Tổng ratio phải = 1.0"

    if labeled_dir is None:
        # Tìm ảnh có nhãn trong scratch hoặc trong dataset/ trực tiếp
        labeled_dir = SCRATCH_DIR

    if not labeled_dir.exists():
        logger.error(f"Thư mục không tồn tại: {labeled_dir}")
        logger.info("→ Chạy 'extract' trước hoặc chỉ định --labeled_dir")
        sys.exit(1)

    # Thu thập ảnh theo video group (key = tên video gốc)
    video_groups: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)

    for img_path in sorted(labeled_dir.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        # Nhãn tương ứng (cùng thư mục, đổi đuôi .txt)
        lbl_path = img_path.with_suffix(".txt")
        if not lbl_path.exists():
            continue  # Bỏ qua ảnh chưa gán nhãn

        # Xác định video group: lấy phần trước '_frame'
        stem  = img_path.stem
        parts = stem.split("_frame")
        vid_key = parts[0] if len(parts) > 1 else stem.rsplit("_", 1)[0]

        video_groups[vid_key].append((img_path, lbl_path))

    if not video_groups:
        logger.warning("Không tìm thấy ảnh có nhãn!")
        logger.info(
            "→ Đảm bảo file .jpg và .txt nằm cùng thư mục "
            f"trong {labeled_dir}"
        )
        return

    # Shuffle danh sách video (không phải frame!)
    video_keys = sorted(video_groups.keys())
    random.seed(seed)
    random.shuffle(video_keys)

    n_total = len(video_keys)
    n_train = max(1, int(round(n_total * ratio[0])))
    n_val   = max(1, int(round(n_total * ratio[1])))
    n_test  = n_total - n_train - n_val

    if n_test < 0:
        n_val  -= 1
        n_test  = n_total - n_train - n_val

    split_map = {
        "train": video_keys[:n_train],
        "val"  : video_keys[n_train : n_train + n_val],
        "test" : video_keys[n_train + n_val :],
    }

    logger.info(
        f"Video-level Split (seed={seed}): "
        f"{n_train} train / {n_val} val / {n_test} test videos"
    )

    # Copy files vào đúng thư mục split
    for split, keys in split_map.items():
        img_out = DATASET / "images" / split
        lbl_out = DATASET / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        total_frames = sum(len(video_groups[k]) for k in keys)
        logger.info(
            f"  {split:5s}: {len(keys)} videos → {total_frames} frames"
        )
        for key in keys:
            for img_src, lbl_src in video_groups[key]:
                shutil.copy2(img_src, img_out / img_src.name)
                shutil.copy2(lbl_src, lbl_out / lbl_src.name)

    logger.info(f"\n✅ Hoàn tất Video-level Split!")
    logger.info("Chạy kiểm tra: python scripts/data_checker.py")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Tổ chức dữ liệu YOLO theo Video-level Split"
    )
    sub = parser.add_subparsers(dest="command")

    # Subcommand: extract
    p_extract = sub.add_parser("extract", help="Trích frame từ video thô")
    p_extract.add_argument(
        "--video_dir", type=Path, default=VIDEOS_RAW,
        help=f"Thư mục video thô (mặc định: {VIDEOS_RAW})"
    )
    p_extract.add_argument(
        "--fps", type=float, default=2.0,
        help="Số frame trích xuất mỗi giây (mặc định: 2)"
    )
    p_extract.add_argument(
        "--max_frames", type=int, default=500,
        help="Số frame tối đa mỗi video (mặc định: 500)"
    )

    # Subcommand: split
    p_split = sub.add_parser("split", help="Chia dữ liệu Video-level")
    p_split.add_argument(
        "--ratio", type=float, nargs=3, default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Tỷ lệ chia (mặc định: 0.70 0.15 0.15)"
    )
    p_split.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (mặc định: 42)"
    )
    p_split.add_argument(
        "--labeled_dir", type=Path, default=None,
        help="Thư mục chứa ảnh đã gán nhãn (mặc định: auto-detect)"
    )

    args = parser.parse_args()

    if args.command == "extract":
        extract_frames(args.video_dir, args.fps, args.max_frames)
    elif args.command == "split":
        video_level_split(
            ratio=tuple(args.ratio),
            seed=args.seed,
            labeled_dir=args.labeled_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

