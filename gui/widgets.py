"""
gui/widgets.py - Cac widget tuy chinh cho giao dien demo
=========================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Nội dung:
    1. VideoLabel        - QLabel hiển thị video và hỗ trợ click chuột vẽ vạch dừng.
    2. ViolationTable    - QTableWidget danh sách vi phạm realtime.
    3. EvidencePanel     - Widget xem 3 ảnh bằng chứng + nút phát clip.
    4. StatsBar          - Thanh thống kê FPS / Tracks / Violations.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore  import Qt, pyqtSignal, QSize
from PyQt6.QtGui   import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QFontDatabase
)
from PyQt6.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QScrollArea, QSizePolicy, QFrame, QAbstractItemView,
)


# ===========================================================================
# VideoLabel — hiển thị video + vẽ vạch bằng click chuột
# ===========================================================================

class VideoLabel(QLabel):
    """
    QLabel mở rộng để:
      - Hiển thị frame video dưới dạng QPixmap.
      - Ghi nhận 2 click chuột trái liên tiếp để xác định đoạn thẳng
        (dùng để cấu hình vạch dừng / vạch xác nhận bằng chuột).
    
    Signals:
        line_drawn(list): Phát [[x1,y1],[x2,y2]] khi người dùng hoàn thành vẽ đoạn.
    """

    line_drawn = pyqtSignal(list)   # [[x1,y1],[x2,y2]] pixel coords trên video gốc

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #111111; border-radius: 6px;")

        self._draw_mode   = False          # Bật/tắt chế độ vẽ vạch
        self._click_pts: List[Tuple[int,int]] = []
        self._orig_size: Optional[Tuple[int,int]] = None  # (w, h) video gốc

        # Giữ 2 vạch cấu hình để vẽ nhánh trên label
        self._stop_line_px: Optional[List[List[int]]] = None
        self._confirm_line_px: Optional[List[List[int]]] = None

    def set_draw_mode(self, enabled: bool) -> None:
        """Bật/tắt chế độ vẽ vạch bằng chuột."""
        self._draw_mode = enabled
        self._click_pts.clear()
        cursor = Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)

    def set_reference_lines(
        self,
        stop_line: Optional[List[List[int]]],
        confirm_line: Optional[List[List[int]]],
    ) -> None:
        """Cập nhật tọa độ vạch để hiển thị trên label (pixel video gốc)."""
        self._stop_line_px    = stop_line
        self._confirm_line_px = confirm_line

    def update_frame(self, jpeg_bytes: bytes, orig_w: int, orig_h: int) -> None:
        """Cập nhật frame mới từ bytes JPEG."""
        self._orig_size = (orig_w, orig_h)
        qimg = QImage.fromData(jpeg_bytes, "JPG")
        if not qimg.isNull():
            pix = QPixmap.fromImage(qimg).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(pix)

    def mousePressEvent(self, event) -> None:
        if not self._draw_mode:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Tính tọa độ điểm click trong hệ tọa độ video gốc
        pt = self._label_to_video(event.position().x(), event.position().y())
        if pt is None:
            return
        self._click_pts.append(pt)

        if len(self._click_pts) == 2:
            line = [list(self._click_pts[0]), list(self._click_pts[1])]
            self.line_drawn.emit(line)
            self._click_pts.clear()

    def _label_to_video(self, lx: float, ly: float) -> Optional[Tuple[int, int]]:
        """Chuyển tọa độ label → tọa độ video gốc, tính đến letterbox."""
        if self._orig_size is None or self.pixmap() is None:
            return None
        pix = self.pixmap()
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        # offset do căn giữa trong label
        ox = (lw - pw) / 2
        oy = (lh - ph) / 2
        # Tọa độ trên pixmap
        px = lx - ox
        py = ly - oy
        if px < 0 or py < 0 or px > pw or py > ph:
            return None
        # Tỉ lệ về video gốc
        vx = int(px / pw * self._orig_size[0])
        vy = int(py / ph * self._orig_size[1])
        return (vx, vy)


# ===========================================================================
# ViolationTable — bảng vi phạm realtime
# ===========================================================================

class ViolationTable(QTableWidget):
    """
    QTableWidget hiển thị danh sách vi phạm theo thời gian thực.

    Signals:
        row_selected(dict): Phát toàn bộ dict thông tin vi phạm khi chọn hàng.
    """

    row_selected = pyqtSignal(dict)

    _HEADERS = ["#", "Thời gian", "ID xe", "Loại xe", "Frame", "Bằng chứng"]
    _WIDTHS   = [30, 90, 60, 100, 65, 80]

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self._HEADERS), parent)
        self.setHorizontalHeaderLabels(self._HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for i, w in enumerate(self._WIDTHS):
            self.setColumnWidth(i, w)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                gridline-color: #313244;
                font-size: 12px;
            }
            QTableWidget::item:selected {
                background-color: #45475a;
            }
            QHeaderView::section {
                background-color: #313244;
                color: #cba6f7;
                padding: 4px;
                border: none;
                font-weight: bold;
            }
            QTableWidget::item:alternate {
                background-color: #181825;
            }
        """)
        self._rows_data: List[dict] = []
        self.cellClicked.connect(self._on_cell_clicked)

    def add_violation(self, ev_dict: dict) -> None:
        """Thêm hàng mới vào bảng vi phạm."""
        import time as _t
        row = self.rowCount()
        self.insertRow(row)
        self._rows_data.append(ev_dict)

        ts_str = _t.strftime("%H:%M:%S", _t.localtime(ev_dict.get("timestamp", 0)))
        has_ev = bool(ev_dict.get("evidence_path", ""))

        values = [
            str(row + 1),
            ts_str,
            f"#{ev_dict.get('track_id', '?')}",
            ev_dict.get("vehicle_class", "?"),
            str(ev_dict.get("frame_index", "?")),
            "✅ Có" if has_ev else "❌ Chưa",
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 0:
                item.setForeground(QColor("#f38ba8"))  # đỏ cho số thứ tự
            self.setItem(row, col, item)

        self.scrollToBottom()

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._rows_data):
            self.row_selected.emit(self._rows_data[row])


# ===========================================================================
# EvidencePanel — xem bằng chứng vi phạm
# ===========================================================================

class EvidencePanel(QWidget):
    """
    Widget hiển thị chi tiết bằng chứng vi phạm:
      - 3 ảnh JPG (trước / cắt vạch / sau).
      - Thông tin metadata (ID, frame, thời gian).
      - Nút mở thư mục bằng chứng.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Tiêu đề
        self._title_lbl = QLabel("Chọn vi phạm để xem bằng chứng")
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl.setStyleSheet("color: #cba6f7; font-weight: bold; font-size: 13px;")
        layout.addWidget(self._title_lbl)

        # 3 ảnh nằm ngang
        img_row = QHBoxLayout()
        self._img_labels: List[QLabel] = []
        captions = ["Trước vi phạm", "Cắt vạch dừng", "Sau vi phạm"]
        for cap_text in captions:
            col = QVBoxLayout()
            lbl = QLabel()
            lbl.setFixedSize(160, 100)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color:#111111; border:1px solid #45475a; border-radius:4px;")
            lbl.setText("—")
            cap_lbl = QLabel(cap_text)
            cap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
            col.addWidget(lbl)
            col.addWidget(cap_lbl)
            img_row.addLayout(col)
            self._img_labels.append(lbl)
        layout.addLayout(img_row)

        # Metadata
        self._meta_lbl = QLabel("")
        self._meta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self._meta_lbl.setWordWrap(True)
        layout.addWidget(self._meta_lbl)

        # Hang nut thao tac: Phat clip MP4 va Mo thu muc bang chung
        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("▶  Phát video vi phạm")
        self._play_btn.setEnabled(False)
        self._play_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; color: #11111b; font-weight: bold;
                border: 1px solid #94e2d5; border-radius: 5px; padding: 5px 14px;
            }
            QPushButton:hover { background-color: #94e2d5; }
            QPushButton:disabled { background-color: #313244; color: #585b70; border-color: #45475a; }
        """)
        self._play_btn.clicked.connect(self._play_clip)
        btn_row.addWidget(self._play_btn)

        self._open_btn = QPushButton("📂  Mở thư mục bằng chứng")
        self._open_btn.setEnabled(False)
        self._open_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 5px; padding: 5px 12px;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton:disabled { color: #585b70; }
        """)
        self._open_btn.clicked.connect(self._open_evidence_dir)
        btn_row.addWidget(self._open_btn)

        layout.addLayout(btn_row)

        self._current_ev_path: str = ""

    def show_evidence(self, ev_dict: dict) -> None:
        """Hiển thị bằng chứng của sự kiện vi phạm đã chọn."""
        import time as _t
        ev_path = ev_dict.get("evidence_path", "")
        self._current_ev_path = ev_path

        ts_str = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(ev_dict.get("timestamp", 0)))
        self._title_lbl.setText(
            f"🚨 Vi phạm ID: {ev_dict.get('violation_id','?')}  |  "
            f"Xe #{ev_dict.get('track_id','?')} ({ev_dict.get('vehicle_class','?')})"
        )
        self._meta_lbl.setText(
            f"Thời gian: {ts_str}  |  "
            f"Frame dừng: {ev_dict.get('stop_line_frame','?')}  |  "
            f"Frame xác nhận: {ev_dict.get('confirm_line_frame','?')}"
        )

        # Load 3 ảnh
        file_names = ["frame_before.jpg", "frame_crossing.jpg", "frame_after.jpg"]
        for lbl, fname in zip(self._img_labels, file_names):
            img_path = str(Path(ev_path) / fname) if ev_path else ""
            if img_path and Path(img_path).exists():
                pix = QPixmap(img_path).scaled(
                    lbl.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                lbl.setPixmap(pix)
            else:
                lbl.setText("Không có ảnh")
                lbl.setPixmap(QPixmap())

        has_dir = bool(ev_path and Path(ev_path).exists())
        clip_path = Path(ev_path) / "clip.mp4" if ev_path else None
        has_clip = bool(clip_path and clip_path.exists())

        self._open_btn.setEnabled(has_dir)
        self._play_btn.setEnabled(has_clip)

    def _play_clip(self) -> None:
        """Phát clip video ngắn 5s ghi nhận khoảnh khắc vi phạm."""
        if self._current_ev_path:
            clip_file = Path(self._current_ev_path) / "clip.mp4"
            if clip_file.exists():
                os.startfile(str(clip_file))

    def _open_evidence_dir(self) -> None:
        if self._current_ev_path and Path(self._current_ev_path).exists():
            os.startfile(self._current_ev_path)


# ===========================================================================
# StatsBar — thanh trạng thái phía dưới
# ===========================================================================

class StatsBar(QWidget):
    """Thanh thống kê nằm ngang: FPS | Tracks | Vi phạm | Đèn | Frame."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet("background-color: #181825; border-top: 1px solid #313244;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(24)

        self._labels: dict = {}
        for key, text in [
            ("fps", "FPS: —"),
            ("tracks", "Tracks: —"),
            ("violations", "Vi phạm: 0"),
            ("light", "Đèn: —"),
            ("frame", "Frame: —"),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #a6adc8; font-size: 12px;")
            layout.addWidget(lbl)
            self._labels[key] = lbl
        layout.addStretch()

    def update_stats(self, stats: dict) -> None:
        self._labels["fps"].setText(f"FPS: {stats.get('fps', 0):.1f}")
        self._labels["tracks"].setText(f"Tracks: {stats.get('tracks', 0)}")
        v = stats.get("violations", 0)
        self._labels["violations"].setText(f"Vi phạm: {v}")
        self._labels["violations"].setStyleSheet(
            "color: #f38ba8; font-weight:bold; font-size:12px;" if v > 0
            else "color: #a6adc8; font-size:12px;"
        )
        self._labels["light"].setText(f"Đèn: {stats.get('light', '—')}")
        fi = stats.get("frame_idx", 0)
        tf = stats.get("total_frames", 0)
        self._labels["frame"].setText(f"Frame: {fi}/{tf}")

