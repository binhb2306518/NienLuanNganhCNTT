"""
scripts/test_pipeline_architecture.py - Kiem tra toan dien kien truc cac module src/
=====================================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ

Mục đích:
    Kiểm tra tính độc lập, không bị lỗi import chéo (Circular Imports),
    và xác thực tính tương thích của các giao diện (Interfaces) giữa 6 module
    chính trong thư mục src/.

Cách chạy:
    cd D:\\NIENLUANNGANH
    .\\venv_nln\\Scripts\\python.exe scripts\\test_pipeline_architecture.py
"""

import sys
import time
from pathlib import Path
import numpy as np

# Thêm thư mục gốc dự án vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("  KIỂM TRA KIẾN TRÚC PIPELINE & GIAO DIỆN CÁC MODULE SRC/")
print("=" * 70)

# ─── 1. KIỂM TRA IMPORT VÀ PHÁT HIỆN IMPORT CHÉO (CIRCULAR IMPORTS) ───────────
print("\n[Bước 1/6] Kiểm tra nạp các module từ package src/...")
try:
    from src import (
        VideoLoader,
        Detector,
        DetectionResult,
        Tracker,
        TrackedObject,
        BaseTracker,
        TrafficLightClassifier,
        LightState,
        TrafficLightResult,
        ViolationDetector,
        ViolationEvent,
        segments_intersect,
        EvidenceExtractor,
    )
    print("  ✓ Import package src/ thành công 100%! Không phát hiện Circular Import.")
except Exception as e:
    print(f"  ❌ LỖI IMPORT: {e}")
    sys.exit(1)

# ─── 2. KIỂM TRA MODULE VIDEOLOADER ───────────────────────────────────────────
print("\n[Bước 2/6] Kiểm tra VideoLoader...")
try:
    loader = VideoLoader(source="dummy_video.mp4", resize_width=1280, fps_limit=30.0)
    assert loader.resize_width == 1280
    assert loader.fps_limit == 30.0
    assert loader.scale_ratio == 1.0
    print(f"  ✓ VideoLoader khởi tạo hợp lệ. Resize target: {loader.resize_width}px.")
except Exception as e:
    print(f"  ❌ Lỗi VideoLoader: {e}")
    sys.exit(1)

# ─── 3. KIỂM TRA MODULE DETECTOR & DETECTIONRESULT ────────────────────────────
print("\n[Bước 3/6] Kiểm tra Detector & DetectionResult...")
try:
    det_res = DetectionResult(
        class_id=1,
        class_name="motorcycle",
        confidence=0.88,
        box=[100, 200, 180, 320],
    )
    assert det_res.center == (140, 260)
    assert det_res.bottom_center == (140, 320)
    assert det_res.width == 80
    assert det_res.height == 120

    detector = Detector(
        model_path="models/pretrained/yolo11s.pt",
        conf_thresh=0.45,
        iou_thresh=0.45,
        device="cuda:0",
        use_fp16=True,
    )
    assert detector.use_fp16 is True
    print(f"  ✓ DetectionResult tính toán đúng tâm ({det_res.center}) và đáy xe ({det_res.bottom_center}).")
    print(f"  ✓ Detector cấu hình đúng FP16 ({detector.use_fp16}) trên thiết bị {detector.device}.")
except Exception as e:
    print(f"  ❌ Lỗi Detector: {e}")
    sys.exit(1)

# ─── 4. KIỂM TRA MODULE TRACKER & STRATEGY PATTERN ────────────────────────────
print("\n[Bước 4/6] Kiểm tra Tracker & hoán đổi ByteTrack / BoT-SORT...")
try:
    tracker_byte = Tracker(algorithm="bytetrack", track_buffer=30)
    tracker_botsort = Tracker(algorithm="botsort", track_buffer=30)

    # Thử nghiệm cập nhật track qua thuật toán heuristic phụ trợ
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracks_f1 = tracker_byte.update(
        dummy_frame, frame_idx=1, detections=[det_res]
    )
    assert len(tracks_f1) == 1
    t1 = tracks_f1[0]
    assert t1.track_id == 1
    assert t1.bottom_center == (140, 320)
    assert len(t1.history) == 1

    # Frame 2: xe di chuyển tới (145, 340)
    det_res2 = DetectionResult(
        class_id=1,
        class_name="motorcycle",
        confidence=0.91,
        box=[105, 215, 185, 340],
    )
    tracks_f2 = tracker_byte.update(
        dummy_frame, frame_idx=2, detections=[det_res2]
    )
    assert tracks_f2[0].track_id == 1
    assert len(tracks_f2[0].history) == 2
    assert tracks_f2[0].velocity == (5.0, 20.0)

    print(f"  ✓ Tracker theo dõi thành công Track #{t1.track_id}. Quỹ đạo tích lũy: {len(tracks_f2[0].history)} điểm.")
    print(f"  ✓ Hoán đổi ByteTrack ↔ BoT-SORT hợp lệ qua tham số cấu hình.")
except Exception as e:
    print(f"  ❌ Lỗi Tracker: {e}")
    sys.exit(1)

# ─── 5. KIỂM TRA MODULE TRAFFIC LIGHT & LOGIC PHÂN TÍCH HSV ──────────────────
print("\n[Bước 5/6] Kiểm tra TrafficLightClassifier & Temporal Smoothing...")
try:
    tl_classifier = TrafficLightClassifier(
        roi=[0, 0, 100, 100],
        min_ratio=0.10,
        smoothing_window=5,
    )

    # Tạo frame giả lập có đốm màu đỏ (BGR: [0, 0, 255])
    red_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    red_frame[20:80, 20:80] = [0, 0, 255]  # Vùng đỏ 60x60 = 36% diện tích

    tl_res = tl_classifier.classify(red_frame)
    assert tl_res.state == LightState.RED
    assert tl_res.confidence > 0.8
    print(f"  ✓ Phân tích HSV nhận dạng đúng trạng thái: {tl_res.state.value} (conf={tl_res.confidence:.2f}).")
    print(f"  ✓ Tỉ lệ màu đo được: {tl_res.color_ratios}")
except Exception as e:
    print(f"  ❌ Lỗi TrafficLightClassifier: {e}")
    sys.exit(1)

# ─── 6. KIỂM TRA VIOLATION LOGIC VÀ EVIDENCE PACKAGING ────────────────────────
print("\n[Bước 6/6] Kiểm tra ViolationDetector & EvidenceExtractor...")
try:
    # Định nghĩa 2 vạch ảo nằm ngang (y=400 và y=450)
    stop_line = [[100, 400], [900, 400]]
    confirm_line = [[100, 450], [900, 450]]

    violation_detector = ViolationDetector(
        stop_line=stop_line,
        confirm_line=confirm_line,
        min_track_len=3,
        red_stable_frames=2,
    )

    # Đưa đèn đỏ vào để thỏa điều kiện ổn định
    violation_detector.update_light(tl_res)
    violation_detector.update_light(tl_res)
    assert violation_detector.is_red_stable is True

    # Tạo đối tượng xe di chuyển xuyên qua cả 2 vạch
    # Quỹ đạo: (300, 350) -> (300, 420) [cắt vạch 1] -> (300, 470) [cắt vạch 2]
    mock_track = TrackedObject(
        track_id=99,
        class_id=1,
        class_name="motorcycle",
        confidence=0.92,
        box=[280, 300, 320, 350],
        bottom_center=(300, 350),
    )
    mock_track.history.append((300, 350))
    mock_track.history.append((300, 380))
    mock_track.history.append((300, 420))  # Cắt Stop Line

    # Đánh giá frame 10 (cắt vạch 1)
    ev1 = violation_detector.check(mock_track, frame_index=10)
    assert ev1 is None  # Chưa vi phạm vì chưa qua Confirmation Line

    # Xe tiến tiếp qua vạch 2
    mock_track.history.append((300, 470))  # Cắt Confirm Line
    ev2 = violation_detector.check(mock_track, frame_index=15)
    assert ev2 is not None  # ĐÃ XÁC NHẬN VI PHẠM!
    assert ev2.track_id == 99
    assert ev2.vehicle_class == "motorcycle"
    assert ev2.status == "CONFIRMED"
    print(f"  ✓ Phát hiện vi phạm thành công: ID={ev2.violation_id}, Xe={ev2.vehicle_class} #{ev2.track_id}.")

    # Kiểm tra đóng gói bằng chứng EvidenceExtractor
    test_evidence_dir = PROJECT_ROOT / "evidence" / "test_scratch"
    extractor = EvidenceExtractor(
        output_dir=test_evidence_dir,
        pre_frames=10,
        post_frames=10,
        fps=30.0,
    )

    # Đẩy một vài frame vào buffer
    for i in range(15):
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        # Vẽ một vòng tròn tượng trưng xe di chuyển
        import cv2
        cv2.circle(f, (300, 350 + i * 10), 15, (0, 255, 0), -1)
        extractor.push_frame(i, f)

    folder = extractor.save(
        event=ev2,
        crossing_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        post_frames_list=[np.zeros((480, 640, 3), dtype=np.uint8)] * 5,
    )

    folder_path = Path(folder)
    assert (folder_path / "frame_before.jpg").exists()
    assert (folder_path / "frame_crossing.jpg").exists()
    assert (folder_path / "frame_after.jpg").exists()
    assert (folder_path / "clip.mp4").exists()
    assert (folder_path / "evidence_info.json").exists()

    print(f"  ✓ EvidenceExtractor đã xuất đầy đủ 5 file chứng cứ tại: {folder_path.name}")
    print("    - frame_before.jpg, frame_crossing.jpg, frame_after.jpg")
    print("    - clip.mp4")
    print("    - evidence_info.json")

    # Dọn dẹp thư mục test scratch
    import shutil
    shutil.rmtree(test_evidence_dir, ignore_errors=True)

except Exception as e:
    print(f"  ❌ Lỗi Violation Logic / Evidence: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("  ✅ TẤT CẢ 6/6 MODULE SRC/ ĐÃ VƯỢT QUA KIỂM TRA TOÀN DIỆN!")
print("  - Giao diện (Interfaces) kết nối chuẩn xác.")
print("  - Không có hiện tượng Circular Import.")
print("  - Cơ chế Strategy Pattern, Geometry Logic và Evidence hoàn toàn thông suốt.")
print("=" * 70)

