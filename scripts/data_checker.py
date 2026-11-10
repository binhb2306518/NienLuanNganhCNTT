"""
data_checker.py — Script kiểm tra toàn vẹn dữ liệu YOLO
=========================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình — B2306518 — CT466E — ĐH Cần Thơ

Chức năng:
  1. Kiểm tra cặp ảnh ↔ nhãn tồn tại đầy đủ
  2. Xác thực format nhãn YOLO (class x_c y_c w h ∈ [0,1])
  3. Phát hiện bounding box suy biến (w/h quá nhỏ, nằm ngoài ảnh)
  4. Thống kê phân bố lớp theo split (train/val/test)
  5. Kiểm tra Data Leakage (trùng filename giữa các split)
  6. Báo cáo tổng hợp ra file report (console + JSON)

Cách dùng:
  cd D:\\NIENLUANNGANH
  .\\venv_nln\\Scripts\\python.exe scripts\\data_checker.py
  .\\venv_nln\\Scripts\\python.exe scripts\\data_checker.py --split train
  .\\venv_nln\\Scripts\\python.exe scripts\\data_checker.py --verbose
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Thêm thư mục gốc vào path để import config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DataChecker")

# ─── Cấu hình dataset ────────────────────────────────────────────────────────
DATASET_ROOT = ROOT / "dataset"
DATA_YAML     = DATASET_ROOT / "data.yaml"
SPLITS        = ["train", "val", "test"]
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 5 lớp đối tượng của dự án
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "traffic_light",
}
NUM_CLASSES = len(CLASS_NAMES)

# Ngưỡng kiểm tra bounding box
MIN_BOX_SIZE   = 0.005   # w hoặc h tối thiểu 0.5% chiều ảnh
MAX_BOX_SIZE   = 1.0     # không được vượt quá 100%
COORD_RANGE    = (0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Class DataChecker
# ═══════════════════════════════════════════════════════════════════════════════
class DataChecker:
    """
    Kiểm tra toàn vẹn tập dữ liệu YOLO cho dự án Niên Luận.

    Attributes
    ----------
    dataset_root : Path
        Thư mục gốc chứa dataset/
    splits : list[str]
        Danh sách split cần kiểm tra
    verbose : bool
        Hiển thị chi tiết từng lỗi
    """

    def __init__(
        self,
        dataset_root: Path = DATASET_ROOT,
        splits: List[str] = None,
        verbose: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.splits       = splits or SPLITS
        self.verbose      = verbose
        self.report: Dict = {
            "timestamp"   : datetime.now().isoformat(),
            "dataset_root": str(self.dataset_root),
            "splits"      : {},
            "leakage"     : {},
            "summary"     : {},
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 1: Thu thập danh sách file
    # ──────────────────────────────────────────────────────────────────────────
    def _collect_files(self, split: str) -> Tuple[List[Path], List[Path]]:
        """Trả về (image_paths, label_paths) cho một split."""
        img_dir = self.dataset_root / "images" / split
        lbl_dir = self.dataset_root / "labels" / split

        if not img_dir.exists():
            logger.warning(f"[{split}] Thư mục ảnh không tồn tại: {img_dir}")
            return [], []
        if not lbl_dir.exists():
            logger.warning(f"[{split}] Thư mục nhãn không tồn tại: {lbl_dir}")
            return [], []

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        labels = sorted(p for p in lbl_dir.iterdir() if p.suffix == ".txt")
        return images, labels

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 2: Kiểm tra cặp ảnh ↔ nhãn
    # ──────────────────────────────────────────────────────────────────────────
    def _check_pairing(
        self, split: str, images: List[Path], labels: List[Path]
    ) -> Dict:
        """Kiểm tra ảnh nào thiếu nhãn và ngược lại."""
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_labels = sorted(img_stems - lbl_stems)
        orphan_labels  = sorted(lbl_stems - img_stems)

        if missing_labels and self.verbose:
            for s in missing_labels[:10]:
                logger.warning(f"  [{split}] Thiếu nhãn cho ảnh: {s}")
        if orphan_labels and self.verbose:
            for s in orphan_labels[:10]:
                logger.warning(f"  [{split}] Nhãn không có ảnh tương ứng: {s}")

        return {
            "images_total"   : len(images),
            "labels_total"   : len(labels),
            "paired"         : len(img_stems & lbl_stems),
            "missing_labels" : missing_labels,
            "orphan_labels"  : orphan_labels,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 3: Xác thực từng file nhãn YOLO
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_label(self, lbl_path: Path) -> Tuple[List[Dict], List[str]]:
        """
        Đọc file nhãn YOLO và trả về:
          - annotations: list[dict] với class_id, x_c, y_c, w, h
          - errors:      list[str] mô tả lỗi

        Format YOLO chuẩn: mỗi dòng = class_id x_center y_center width height
        Tất cả tọa độ normalize [0, 1] theo kích thước ảnh.
        """
        annotations = []
        errors      = []

        try:
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        except Exception as e:
            errors.append(f"Không đọc được file: {e}")
            return annotations, errors

        # File nhãn rỗng = ảnh không có đối tượng → hợp lệ (background)
        if not lines:
            return annotations, errors

        for i, line in enumerate(lines):
            parts = line.strip().split()
            lineno = i + 1

            # Kiểm tra số trường
            if len(parts) != 5:
                errors.append(
                    f"Dòng {lineno}: cần 5 trường, nhận {len(parts)} → '{line}'"
                )
                continue

            # Parse
            try:
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"Dòng {lineno}: không parse được số → '{line}'")
                continue

            # Kiểm tra class id hợp lệ
            if cls_id not in CLASS_NAMES:
                errors.append(
                    f"Dòng {lineno}: class_id={cls_id} không hợp lệ "
                    f"(phải trong 0–{NUM_CLASSES - 1})"
                )

            # Kiểm tra tọa độ nằm trong [0, 1]
            for name, val in [("x_c", x_c), ("y_c", y_c), ("w", w), ("h", h)]:
                if not (COORD_RANGE[0] <= val <= COORD_RANGE[1]):
                    errors.append(
                        f"Dòng {lineno}: {name}={val:.4f} nằm ngoài [0, 1]"
                    )

            # Kiểm tra bounding box không suy biến (quá nhỏ)
            if w < MIN_BOX_SIZE:
                errors.append(
                    f"Dòng {lineno}: w={w:.4f} quá nhỏ (< {MIN_BOX_SIZE}) "
                    "— có thể lỗi nhãn hoặc pixel noise"
                )
            if h < MIN_BOX_SIZE:
                errors.append(
                    f"Dòng {lineno}: h={h:.4f} quá nhỏ (< {MIN_BOX_SIZE}) "
                    "— có thể lỗi nhãn hoặc pixel noise"
                )

            # Kiểm tra box vẫn nằm trong ảnh (x_c ± w/2 ∈ [0,1])
            x_min = x_c - w / 2
            x_max = x_c + w / 2
            y_min = y_c - h / 2
            y_max = y_c + h / 2
            if x_min < -0.01 or x_max > 1.01 or y_min < -0.01 or y_max > 1.01:
                errors.append(
                    f"Dòng {lineno}: bounding box tràn ra ngoài ảnh "
                    f"[{x_min:.3f}, {x_max:.3f}] × [{y_min:.3f}, {y_max:.3f}]"
                )

            if cls_id in CLASS_NAMES:
                annotations.append(
                    {
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES[cls_id],
                        "x_c": x_c,
                        "y_c": y_c,
                        "w": w,
                        "h": h,
                        "area": w * h,
                    }
                )

        return annotations, errors

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 4: Thống kê phân bố lớp
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_stats(self, all_annotations: List[Dict]) -> Dict:
        """Tính phân bố lớp, diện tích box trung bình."""
        class_counts  = defaultdict(int)
        class_areas   = defaultdict(list)

        for ann in all_annotations:
            cname = ann["class_name"]
            class_counts[cname]  += 1
            class_areas[cname].append(ann["area"])

        total = sum(class_counts.values())
        stats = {}
        for cls_name in CLASS_NAMES.values():
            count = class_counts.get(cls_name, 0)
            areas = class_areas.get(cls_name, [0])
            avg_area = sum(areas) / len(areas) if areas else 0.0
            stats[cls_name] = {
                "count"   : count,
                "percent" : round(100 * count / total, 2) if total > 0 else 0.0,
                "avg_area": round(avg_area, 5),
            }

        return {"total_annotations": total, "class_distribution": stats}

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 5: Kiểm tra Data Leakage
    # ──────────────────────────────────────────────────────────────────────────
    def _check_leakage(self, split_stems: Dict[str, set]) -> Dict:
        """
        So sánh chéo tất cả cặp split để phát hiện filename trùng lặp.
        Video-level split đúng không được có frame trùng giữa các split.
        """
        leakage = {}
        split_list = list(split_stems.items())

        for i in range(len(split_list)):
            for j in range(i + 1, len(split_list)):
                split_a, stems_a = split_list[i]
                split_b, stems_b = split_list[j]
                overlap = stems_a & stems_b
                key = f"{split_a}↔{split_b}"
                leakage[key] = {
                    "overlap_count": len(overlap),
                    "examples"     : sorted(overlap)[:5],
                }
                if overlap:
                    logger.error(
                        f"⚠ DATA LEAKAGE: {len(overlap)} file trùng giữa "
                        f"{split_a} và {split_b}: {list(overlap)[:3]}"
                    )
                else:
                    logger.info(f"  ✓ Không rò rỉ dữ liệu: {split_a} ↔ {split_b}")

        return leakage

    # ──────────────────────────────────────────────────────────────────────────
    # Main: Chạy kiểm tra
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> Dict:
        """Chạy toàn bộ pipeline kiểm tra, trả về report dict."""
        logger.info("=" * 65)
        logger.info("  DATA CHECKER — Dự án Niên Luận NLN")
        logger.info(f"  Dataset root: {self.dataset_root}")
        logger.info("=" * 65)

        split_stems: Dict[str, set] = {}
        total_errors = 0
        total_annotations_all = 0

        for split in self.splits:
            logger.info(f"\n{'─'*50}")
            logger.info(f"  SPLIT: {split.upper()}")
            logger.info(f"{'─'*50}")

            images, labels = self._collect_files(split)

            if not images and not labels:
                logger.info(f"  [SKIP] Split '{split}' rỗng (chưa có dữ liệu)")
                self.report["splits"][split] = {"status": "empty"}
                split_stems[split] = set()
                continue

            # 2. Kiểm tra cặp
            pairing = self._check_pairing(split, images, labels)
            logger.info(
                f"  Ảnh: {pairing['images_total']} | "
                f"Nhãn: {pairing['labels_total']} | "
                f"Cặp hợp lệ: {pairing['paired']}"
            )
            if pairing["missing_labels"]:
                logger.warning(
                    f"  ⚠ Thiếu nhãn: {len(pairing['missing_labels'])} ảnh"
                )
            if pairing["orphan_labels"]:
                logger.warning(
                    f"  ⚠ Nhãn mồ côi: {len(pairing['orphan_labels'])} file"
                )

            # 3. Validate từng nhãn
            lbl_dir = self.dataset_root / "labels" / split
            all_annotations: List[Dict] = []
            split_errors: Dict[str, List[str]] = {}
            split_errors_count = 0

            for img_path in images:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue
                anns, errs = self._validate_label(lbl_path)
                all_annotations.extend(anns)
                if errs:
                    split_errors[img_path.name] = errs
                    split_errors_count += len(errs)
                    if self.verbose:
                        for err in errs:
                            logger.warning(f"    {img_path.name}: {err}")

            if split_errors_count == 0:
                logger.info("  ✓ Tất cả nhãn hợp lệ — không có lỗi format")
            else:
                logger.warning(
                    f"  ⚠ Phát hiện {split_errors_count} lỗi "
                    f"trong {len(split_errors)} file nhãn"
                )
            total_errors += split_errors_count

            # 4. Thống kê lớp
            stats = self._compute_stats(all_annotations)
            total_annotations_all += stats["total_annotations"]
            logger.info(
                f"  Tổng annotations: {stats['total_annotations']}"
            )
            logger.info("  Phân bố lớp:")
            for cls_name, info in stats["class_distribution"].items():
                bar = "█" * min(int(info["percent"] / 2), 25)
                logger.info(
                    f"    {cls_name:<14} {info['count']:>6} "
                    f"({info['percent']:>5.1f}%) {bar}"
                )

            split_stems[split] = {p.stem for p in images}
            self.report["splits"][split] = {
                "pairing"       : pairing,
                "label_errors"  : split_errors,
                "error_count"   : split_errors_count,
                "statistics"    : stats,
            }

        # 5. Kiểm tra Data Leakage
        logger.info(f"\n{'─'*50}")
        logger.info("  KIỂM TRA DATA LEAKAGE")
        logger.info(f"{'─'*50}")
        self.report["leakage"] = self._check_leakage(split_stems)

        # 6. Tổng kết
        logger.info(f"\n{'═'*65}")
        logger.info("  KẾT QUẢ TỔNG HỢP")
        logger.info(f"{'═'*65}")

        all_empty = all(
            self.report["splits"].get(s, {}).get("status") == "empty"
            for s in self.splits
        )

        if all_empty:
            logger.info(
                "  ⚠ Dataset chưa có dữ liệu — Thư mục images/ và labels/ rỗng."
            )
            logger.info(
                "  → Hướng dẫn: Thu thập video → Trích frame → Gán nhãn → "
                "Chia Video-level split."
            )
            status = "EMPTY_DATASET"
        elif total_errors == 0:
            logger.info("  ✅ Dataset hợp lệ — Không phát hiện lỗi nào!")
            status = "OK"
        else:
            logger.warning(
                f"  ⚠ Dataset có {total_errors} lỗi cần sửa trước khi training!"
            )
            status = "HAS_ERRORS"

        self.report["summary"] = {
            "status"             : status,
            "total_errors"       : total_errors,
            "total_annotations"  : total_annotations_all,
        }

        return self.report

    # ──────────────────────────────────────────────────────────────────────────
    # Xuất report ra JSON
    # ──────────────────────────────────────────────────────────────────────────
    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """Lưu report ra file JSON trong logs/."""
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = ROOT / "logs" / f"data_check_{ts}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        logger.info(f"\n  📄 Report đã lưu: {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Kiểm tra toàn vẹn tập dữ liệu YOLO cho dự án NLN"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Chỉ kiểm tra split cụ thể (mặc định: all)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_ROOT,
        help=f"Đường dẫn tới dataset/ (mặc định: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị chi tiết từng lỗi",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Không lưu file report JSON",
    )
    return parser.parse_args()


def main():
    args  = parse_args()
    splits = SPLITS if args.split == "all" else [args.split]

    checker = DataChecker(
        dataset_root=args.dataset,
        splits=splits,
        verbose=args.verbose,
    )
    report = checker.run()

    if not args.no_report:
        checker.save_report()

    # Exit code: 0 nếu OK hoặc EMPTY, 1 nếu có lỗi
    status = report["summary"].get("status", "UNKNOWN")
    sys.exit(0 if status in ("OK", "EMPTY_DATASET") else 1)


if __name__ == "__main__":
    main()

