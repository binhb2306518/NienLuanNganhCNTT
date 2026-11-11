"""
main.py - Diem khoi dong ung dung Demo Phat hien Vi pham Den Do
===============================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Cách chạy:
    python main.py

Yêu cầu:
    - PyQt6 >= 6.5          pip install PyQt6
    - Ultralytics >= 8.4    pip install ultralytics
    - OpenCV-python >= 4.8  pip install opencv-python
    - PyYAML                pip install pyyaml
    - GPU NVIDIA + CUDA 12.x (tùy chọn, tự động fallback sang CPU)

Kiến trúc giao diện (3 panel):
    LEFT   : VideoLabel hiển thị video + overlay vạch + bounding box
    RIGHT  : ViolationTable (trên) + EvidencePanel (dưới)
    BOTTOM : StatsBar + StatusBar
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình logging trước khi import bất kỳ thứ gì
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "logs" / "app.log",
            mode="a",
            encoding="utf-8",
        ) if (Path(__file__).parent / "logs").exists()
          else logging.NullHandler(),
    ],
)

logger = logging.getLogger("Main")

# ---------------------------------------------------------------------------
# Kiểm tra PyQt6
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore    import Qt
    from PyQt6.QtGui     import QFont
except ImportError:
    print(
        "[LỖI] PyQt6 chưa được cài đặt!\n"
        "       Chạy lệnh: pip install PyQt6\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import MainWindow
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from gui.main_window import MainWindow


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # Bật High-DPI scaling (hỗ trợ màn hình 4K / 2K)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("TrafVio - He thong Phat hien Vi pham Den Do")
    app.setOrganizationName("ĐH Cần Thơ · CT466E")
    app.setApplicationVersion("1.0.0")

    # Font mặc định
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Tạo và hiển thị cửa sổ chính
    window = MainWindow()
    window.show()

    logger.info("=" * 60)
    logger.info("ỨNG DỤNG DEMO PHÁT HIỆN VI PHẠM ĐÈN ĐỎ")
    logger.info("Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("Hướng dẫn sử dụng:")
    logger.info("  1. Click [Chọn video] → chọn file MP4/AVI")
    logger.info("  2. (Tùy chọn) Click [Vẽ vạch dừng] và [Vẽ vạch xác nhận]")
    logger.info("     để định nghĩa lại vạch trực tiếp trên khung video.")
    logger.info("  3. Click [Bắt đầu] → pipeline AI tự động chạy.")
    logger.info("  4. Click vào hàng trong bảng vi phạm để xem ảnh bằng chứng.")
    logger.info("=" * 60)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

