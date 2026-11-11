"""
gui/main_window.py - Cua so chinh PyQt6
========================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Bố cục giao diện:
  ┌──────────────────────────────────────────────────────────────┐
  │  TOOLBAR: [Chọn video] [Bắt đầu] [Tạm dừng] [Vẽ vạch dừng]│
  │          [Vẽ vạch xác nhận] [Dừng]                          │
  ├────────────────────────────┬─────────────────────────────────┤
  │   VIDEO DISPLAY (trái)     │  PANEL PHẢI:                    │
  │   VideoLabel (60% width)   │  ├─ Danh sách vi phạm          │
  │                            │  └─ Chi tiết bằng chứng         │
  ├────────────────────────────┴─────────────────────────────────┤
  │  STATS BAR: FPS | Tracks | Vi phạm | Đèn | Frame            │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore  import Qt, QTimer
from PyQt6.QtGui   import QIcon, QAction, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QPushButton, QFileDialog,
    QMessageBox, QLabel, QGroupBox, QStatusBar,
    QSizePolicy, QFrame,
)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from gui.widgets        import VideoLabel, ViolationTable, EvidencePanel, StatsBar
from gui.pipeline_worker import PipelineWorker

logger = logging.getLogger("MainWindow")

# ---------------------------------------------------------------------------
# Cấu hình mặc định (đọc từ config.yaml nếu có, fallback sang hardcode)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL    = str(PROJECT_ROOT / "models" / "pretrained" / "yolov8s.pt")
_DEFAULT_EVIDENCE = str(PROJECT_ROOT / "evidence")
_DEFAULT_STOP     = [[100, 400], [900, 400]]
_DEFAULT_CONFIRM  = [[100, 450], [900, 450]]

# Cố gắng đọc config.yaml
try:
    import yaml
    _cfg_path = PROJECT_ROOT / "config.yaml"
    if _cfg_path.exists():
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f)
        _DEFAULT_MODEL    = _cfg.get("model", {}).get("detector", _DEFAULT_MODEL)
        _DEFAULT_EVIDENCE = _cfg.get("paths", {}).get("evidence_dir", _DEFAULT_EVIDENCE)
        _sl = _cfg.get("stop_line", {})
        _DEFAULT_STOP    = _sl.get("stop_line_coords",    _DEFAULT_STOP)
        _DEFAULT_CONFIRM = _sl.get("confirm_line_coords", _DEFAULT_CONFIRM)
except Exception:
    pass


# ===========================================================================
# STYLE SHEET toàn ứng dụng (Catppuccin Mocha dark theme)
# ===========================================================================

APP_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 6px;
    padding: 4px 8px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 14px;
    min-width: 80px;
}
QPushButton:hover  { background-color: #45475a; }
QPushButton:pressed{ background-color: #585b70; }
QPushButton:disabled { color: #585b70; border-color: #313244; }
QPushButton#btn_start  { background-color: #40a02b; color: #fff; border-color: #40a02b; }
QPushButton#btn_start:hover  { background-color: #4caf31; }
QPushButton#btn_pause  { background-color: #fe640b; color: #fff; border-color: #fe640b; }
QPushButton#btn_pause:hover  { background-color: #ff7a28; }
QPushButton#btn_stop   { background-color: #d20f39; color: #fff; border-color: #d20f39; }
QPushButton#btn_stop:hover   { background-color: #e21045; }
QPushButton#btn_draw_stop {
    background-color: #313244; border-color: #f38ba8; color: #f38ba8;
}
QPushButton#btn_draw_confirm {
    background-color: #313244; border-color: #f9e2af; color: #f9e2af;
}
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    color: #cba6f7;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QSplitter::handle {
    background-color: #313244;
    width: 3px;
}
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    font-size: 11px;
}
QLabel#mode_lbl {
    color: #f9e2af;
    font-weight: bold;
    padding: 0 8px;
}
"""


class MainWindow(QMainWindow):
    """
    Cửa sổ chính của ứng dụng demo phát hiện vi phạm đèn đỏ.

    Luồng giao tiếp:
        UI Thread (MainWindow) ← Qt Signals ← PipelineWorker (QThread)
    
    Người dùng có thể:
        - Chọn file video và chạy pipeline.
        - Tạm dừng / tiếp tục / dừng pipeline.
        - Click chuột để vẽ vạch dừng và vạch xác nhận trực tiếp trên video.
        - Xem danh sách vi phạm theo thời gian thực.
        - Click vào vi phạm để xem 3 ảnh bằng chứng.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "Hệ Thống Phát Hiện Vi Phạm Đèn Tín Hiệu Giao Thông  —  "
            "Nguyễn Thanh Bình · B2306518 · CT466E · ĐH Cần Thơ"
        )
        self.resize(1280, 760)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(APP_STYLE)

        # Cấu hình vạch
        self._stop_line:    List[List[int]] = [list(p) for p in _DEFAULT_STOP]
        self._confirm_line: List[List[int]] = [list(p) for p in _DEFAULT_CONFIRM]
        self._draw_target: Optional[str] = None   # "stop" | "confirm" | None

        # Worker
        self._worker: Optional[PipelineWorker] = None
        self._video_path  = ""
        self._model_path  = _DEFAULT_MODEL
        self._video_w     = 1280
        self._video_h     = 720

        # Build UI
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._update_line_display()

    # ------------------------------------------------------------------
    # BUILD UI
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("Điều khiển", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        # Chọn video
        self._btn_open = QPushButton("📂  Chọn video")
        self._btn_open.setObjectName("btn_open")
        self._btn_open.clicked.connect(self._on_open_video)
        tb.addWidget(self._btn_open)

        tb.addSeparator()

        # Bắt đầu
        self._btn_start = QPushButton("▶  Bắt đầu")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start)
        tb.addWidget(self._btn_start)

        # Tạm dừng
        self._btn_pause = QPushButton("⏸  Tạm dừng")
        self._btn_pause.setObjectName("btn_pause")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._on_pause)
        tb.addWidget(self._btn_pause)

        # Dừng
        self._btn_stop = QPushButton("⏹  Dừng")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        tb.addWidget(self._btn_stop)

        tb.addSeparator()

        # Vẽ vạch dừng
        self._btn_draw_stop = QPushButton("✏ Vạch dừng")
        self._btn_draw_stop.setObjectName("btn_draw_stop")
        self._btn_draw_stop.setCheckable(True)
        self._btn_draw_stop.setToolTip(
            "Click lần 1: điểm đầu  |  Click lần 2: điểm cuối → vẽ Vạch Dừng (đỏ)"
        )
        self._btn_draw_stop.clicked.connect(lambda: self._enter_draw_mode("stop"))
        tb.addWidget(self._btn_draw_stop)

        # Vẽ vạch xác nhận
        self._btn_draw_confirm = QPushButton("✏ Vạch xác nhận")
        self._btn_draw_confirm.setObjectName("btn_draw_confirm")
        self._btn_draw_confirm.setCheckable(True)
        self._btn_draw_confirm.setToolTip(
            "Click lần 1 & 2 → vẽ Vạch Xác Nhận (vàng)"
        )
        self._btn_draw_confirm.clicked.connect(lambda: self._enter_draw_mode("confirm"))
        tb.addWidget(self._btn_draw_confirm)

        # Nhãn chế độ vẽ
        self._mode_lbl = QLabel("")
        self._mode_lbl.setObjectName("mode_lbl")
        tb.addWidget(self._mode_lbl)

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 6, 8, 4)
        main_layout.setSpacing(6)

        # Splitter ngang: video (trái) | panel phải
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # --- Panel trái: Video ---
        left_group = QGroupBox("📹  Luồng video")
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(4, 10, 4, 4)

        self._video_lbl = VideoLabel()
        self._video_lbl.line_drawn.connect(self._on_line_drawn)
        left_layout.addWidget(self._video_lbl)

        # Thông tin tọa độ vạch
        self._line_info_lbl = QLabel()
        self._line_info_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self._line_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self._line_info_lbl)

        splitter.addWidget(left_group)

        # --- Panel phải ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Bảng vi phạm
        vio_group = QGroupBox("🚨  Danh sách vi phạm")
        vio_layout = QVBoxLayout(vio_group)
        vio_layout.setContentsMargins(4, 10, 4, 4)
        self._vio_table = ViolationTable()
        self._vio_table.row_selected.connect(self._on_violation_selected)
        vio_layout.addWidget(self._vio_table)
        right_layout.addWidget(vio_group, stretch=1)

        # Bằng chứng
        ev_group = QGroupBox("🔍  Chi tiết bằng chứng")
        ev_layout = QVBoxLayout(ev_group)
        ev_layout.setContentsMargins(4, 10, 4, 4)
        self._ev_panel = EvidencePanel()
        ev_layout.addWidget(self._ev_panel)
        right_layout.addWidget(ev_group, stretch=0)

        splitter.addWidget(right_panel)
        splitter.setSizes([820, 440])

        main_layout.addWidget(splitter, stretch=1)

        # Stats bar
        self._stats_bar = StatsBar()
        main_layout.addWidget(self._stats_bar)

    def _build_statusbar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Sẵn sàng  |  Mở file video để bắt đầu  |  "
            f"Stop Line: {self._stop_line}   Confirm Line: {self._confirm_line}"
        )

    # ------------------------------------------------------------------
    # CẬP NHẬT HIỂN THỊ
    # ------------------------------------------------------------------

    def _update_line_display(self) -> None:
        sl = self._stop_line
        cl = self._confirm_line
        self._line_info_lbl.setText(
            f"Vạch dừng: [{sl[0][0]},{sl[0][1]}]→[{sl[1][0]},{sl[1][1]}]   |   "
            f"Vạch xác nhận: [{cl[0][0]},{cl[0][1]}]→[{cl[1][0]},{cl[1][1]}]"
        )
        self._video_lbl.set_reference_lines(self._stop_line, self._confirm_line)
        self._status.showMessage(
            f"Stop Line: {self._stop_line}   |   Confirm Line: {self._confirm_line}"
        )

    # ------------------------------------------------------------------
    # XỬ LÝ SỰ KIỆN TOOLBAR
    # ------------------------------------------------------------------

    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file video",
            str(PROJECT_ROOT / "dataset"),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self._video_path = path
            self._btn_start.setEnabled(True)
            self._status.showMessage(f"Đã chọn: {path}")
            # Thử đọc kích thước video để hiển thị placeholder
            import cv2 as _cv2
            cap = _cv2.VideoCapture(path)
            if cap.isOpened():
                self._video_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                self._video_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                fps   = cap.get(_cv2.CAP_PROP_FPS)
                cap.release()
                self._status.showMessage(
                    f"Video: {Path(path).name}  |  "
                    f"{self._video_w}×{self._video_h}  |  "
                    f"{fps:.0f}fps  |  {total} frames"
                )

    def _on_start(self) -> None:
        if not self._video_path:
            QMessageBox.warning(self, "Chưa chọn video", "Vui lòng chọn file video trước.")
            return
        if not Path(self._model_path).exists():
            QMessageBox.critical(
                self, "Không tìm thấy model",
                f"Không tìm thấy model tại:\n{self._model_path}\n\n"
                "Vui lòng tải model YOLO về thư mục models/pretrained/."
            )
            return

        # Nếu worker đang chạy, dừng trước
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)

        self._worker = PipelineWorker(
            video_path   = self._video_path,
            model_path   = self._model_path,
            stop_line    = self._stop_line,
            confirm_line = self._confirm_line,
            evidence_dir = _DEFAULT_EVIDENCE,
            red_stable_frames = 15,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.violation_found.connect(self._on_violation_found)
        self._worker.stats_updated.connect(self._stats_bar.update_stats)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error_occurred.connect(self._on_error)

        self._worker.start()

        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_open.setEnabled(False)
        self._status.showMessage("🔄  Pipeline AI đang chạy...")

    def _on_pause(self) -> None:
        if self._worker is None:
            return
        if self._btn_pause.text().startswith("⏸"):
            self._worker.pause()
            self._btn_pause.setText("▶  Tiếp tục")
            self._status.showMessage("⏸  Đã tạm dừng")
        else:
            self._worker.resume()
            self._btn_pause.setText("⏸  Tạm dừng")
            self._status.showMessage("🔄  Đang chạy...")

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
        self._btn_pause.setText("⏸  Tạm dừng")

    def _on_pipeline_finished(self) -> None:
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_open.setEnabled(True)
        n = self._vio_table.rowCount()
        self._status.showMessage(
            f"✅  Hoàn tất  |  Tổng vi phạm: {n}  |  Bằng chứng lưu tại: {_DEFAULT_EVIDENCE}"
        )

    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Lỗi Pipeline", f"Pipeline AI báo lỗi:\n\n{msg}")
        self._on_pipeline_finished()

    # ------------------------------------------------------------------
    # XỬ LÝ TÍN HIỆU TỪ WORKER
    # ------------------------------------------------------------------

    def _on_frame_ready(self, jpeg_bytes: bytes, fps: float) -> None:
        self._video_lbl.update_frame(jpeg_bytes, self._video_w, self._video_h)

    def _on_violation_found(self, ev_dict: dict) -> None:
        self._vio_table.add_violation(ev_dict)
        self._status.showMessage(
            f"🚨  Vi phạm mới: Track #{ev_dict.get('track_id')} "
            f"({ev_dict.get('vehicle_class')}) tại frame {ev_dict.get('frame_index')}"
        )

    def _on_violation_selected(self, ev_dict: dict) -> None:
        self._ev_panel.show_evidence(ev_dict)

    # ------------------------------------------------------------------
    # VẼ VẠCH BẰNG CHUỘT
    # ------------------------------------------------------------------

    def _enter_draw_mode(self, target: str) -> None:
        """Bật chế độ vẽ cho vạch dừng ('stop') hoặc vạch xác nhận ('confirm')."""
        if self._draw_target == target:
            # Toggle off
            self._draw_target = None
            self._video_lbl.set_draw_mode(False)
            self._btn_draw_stop.setChecked(False)
            self._btn_draw_confirm.setChecked(False)
            self._mode_lbl.setText("")
            return

        self._draw_target = target
        self._video_lbl.set_draw_mode(True)

        if target == "stop":
            self._btn_draw_stop.setChecked(True)
            self._btn_draw_confirm.setChecked(False)
            self._mode_lbl.setText("✏  Đang vẽ VẠCH DỪNG  —  Click điểm 1 rồi điểm 2")
        else:
            self._btn_draw_confirm.setChecked(True)
            self._btn_draw_stop.setChecked(False)
            self._mode_lbl.setText("✏  Đang vẽ VẠCH XÁC NHẬN  —  Click điểm 1 rồi điểm 2")

    def _on_line_drawn(self, line: list) -> None:
        """Nhận tọa độ vạch vừa vẽ từ VideoLabel."""
        if self._draw_target == "stop":
            self._stop_line = line
        elif self._draw_target == "confirm":
            self._confirm_line = line

        # Tắt chế độ vẽ sau khi hoàn thành
        self._draw_target = None
        self._video_lbl.set_draw_mode(False)
        self._btn_draw_stop.setChecked(False)
        self._btn_draw_confirm.setChecked(False)
        self._mode_lbl.setText("")

        self._update_line_display()
        logger.info(f"Vạch cập nhật: Stop={self._stop_line}  Confirm={self._confirm_line}")

    # ------------------------------------------------------------------
    # ĐÓNG CỬA SỔ
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(5000)
        event.accept()

