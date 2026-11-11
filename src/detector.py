"""
src/detector.py - Wrapper YOLO cho phat hien phuong tien va den tin hieu
=========================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Bao bọc (wrapper) mô hình Ultralytics YOLO (hỗ trợ YOLOv8 và YOLO11).
    Tích hợp cơ chế bật/tắt bán chính xác FP16 (Automatic Mixed Precision),
    kiểm soát ngưỡng tin cậy, ngưỡng NMS IoU, và hỗ trợ trích xuất tọa độ
    tâm đáy (bottom center) mô phỏng tiếp điểm bánh xe với mặt đường.

Input:
    - Frame ảnh numpy.ndarray (H, W, 3) định dạng BGR.

Output:
    - List[DetectionResult]: Danh sách đối tượng phát hiện với bounding box,
      lớp phân loại, độ tin cậy và tọa độ hình học chuẩn hóa.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("Detector")


@dataclass
class DetectionResult:
    """
    Cấu trúc dữ liệu chứa kết quả phát hiện một đối tượng trên khung hình.

    Attributes:
        class_id (int): Mã định danh lớp (0=car, 1=motorcycle, 2=bus, 3=truck, 4=traffic_light).
        class_name (str): Tên chuỗi đại diện cho lớp.
        confidence (float): Độ tin cậy dự đoán nằm trong khoảng [0.0, 1.0].
        box (List[int]): Tọa độ hộp bao [x1, y1, x2, y2] tính theo pixel.
        center (Tuple[int, int]): Tọa độ tâm hình học (cx, cy) của hộp bao.
        bottom_center (Tuple[int, int]): Tọa độ trung điểm cạnh đáy hộp bao (cx, y2),
            dùng làm điểm đại diện tiếp xúc mặt đường của phương tiện khi xét cắt vạch dừng.
    """
    class_id: int
    class_name: str
    confidence: float
    box: List[int]
    center: Tuple[int, int] = field(init=False)
    bottom_center: Tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        """Tự động tính toán tâm hình học và điểm tiếp đất (bottom-center)."""
        x1, y1, x2, y2 = self.box
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.bottom_center = ((x1 + x2) // 2, y2)

    @property
    def width(self) -> int:
        """Chiều rộng bounding box."""
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        """Chiều cao bounding box."""
        return max(0, self.box[3] - self.box[1])

    @property
    def area(self) -> int:
        """Diện tích bounding box."""
        return self.width * self.height


class Detector:
    """
    Lớp điều khiển suy luận phát hiện đối tượng dựa trên Ultralytics YOLO.

    Tối ưu hóa đặc thù cho Laptop GPU RTX 3050 6GB:
    - Tự động bật FP16 khi chạy trên CUDA để giảm 50% băng thông VRAM.
    - Tự động fallback về FP32 nếu chạy trên CPU (tránh lỗi PyTorch half tensor).
    - Cung cấp warmup khởi động để làm nóng CUDA runtime.

    Attributes:
        model_path (str): Đường dẫn tới trọng số .pt của YOLO.
        conf_thresh (float): Ngưỡng tin cậy lọc kết quả (Confidence Threshold).
        iou_thresh (float): Ngưỡng IoU phục vụ giải thuật NMS.
        device (str): Thiết bị tính toán ('cuda:0' hoặc 'cpu').
        use_fp16 (bool): Bật nửa chính xác FP16 (chỉ hợp lệ khi device có cuda).
        classes (Optional[List[int]]): Danh sách các lớp cần lọc.
    """

    CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "traffic_light"]

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo11s.pt",
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.45,
        device: str = "cuda:0",
        use_fp16: bool = True,
        classes: Optional[List[int]] = None,
    ) -> None:
        """
        Khởi tạo detector với các tham số suy luận.
        """
        self.model_path = str(model_path)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.device = str(device)
        self.classes = classes

        # Bảo vệ: Nếu chọn CPU thì bắt buộc tắt FP16 để tránh crash PyTorch
        if "cpu" in self.device.lower():
            self.use_fp16 = False
            if use_fp16:
                logger.warning("Đã tự động tắt FP16 vì thiết bị là CPU.")
        else:
            self.use_fp16 = bool(use_fp16)

        self._model = None

    def load(self) -> None:
        """
        Tải mô hình YOLO vào bộ nhớ thiết bị tính toán.
        Thực hiện bước warm-up nhẹ để nạp CUDA kernel vào VRAM.
        """
        from ultralytics import YOLO
        logger.info(f"Đang nạp mô hình YOLO từ: {self.model_path} lên {self.device}...")
        self._model = YOLO(self.model_path)

        # Chuyển thiết bị
        try:
            self._model.to(self.device)
        except Exception as e:
            logger.warning(f"Không thể chuyển model lên {self.device}: {e}. Dùng mặc định.")

        # Warm-up GPU bằng dummy frame
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(
                source=dummy,
                device=self.device,
                half=self.use_fp16,
                verbose=False,
            )
            logger.info("Hoàn tất warm-up GPU cho mô hình Detector.")
        except Exception as e:
            logger.debug(f"Bỏ qua bước warm-up (không ảnh hưởng suy luận): {e}")

    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Thực hiện suy luận phát hiện đối tượng trên một khung hình ảnh.

        Args:
            frame: Ảnh mảng numpy BGR kích thước (H, W, 3).

        Returns:
            List[DetectionResult]: Danh sách các bounding box hợp lệ sau lọc.
        """
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            half=self.use_fp16,
            classes=self.classes,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results:
            return detections

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]

                # Xác định tên lớp an toàn
                if cls_id < len(self.CLASS_NAMES):
                    cname = self.CLASS_NAMES[cls_id]
                elif r.names and cls_id in r.names:
                    cname = r.names[cls_id]
                else:
                    cname = str(cls_id)

                detections.append(
                    DetectionResult(
                        class_id=cls_id,
                        class_name=cname,
                        confidence=conf,
                        box=xyxy,
                    )
                )

        return detections

    @property
    def is_loaded(self) -> bool:
        """Kiểm tra trạng thái nạp của mô hình."""
        return self._model is not None

    @property
    def raw_model(self):
        """Truy xuất đối tượng mô hình YOLO gốc bên trong."""
        return self._model
