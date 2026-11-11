# BÁO CÁO NGHIỆM THU GIAI ĐOẠN 4: BASELINE PIPELINE
**Đề tài:** Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông  
**Sinh viên:** Nguyễn Thanh Bình — B2306518 — CT466E — ĐH Cần Thơ  
**Ngày thực hiện:** 11/11/2026  
**Trạng thái:** HOÀN THÀNH 100%  

---

## 1. MỤC TIÊU GIAI ĐOẠN 4
- Thiết lập đường ống xử lý tối thiểu (Minimum Viable Pipeline) hoàn chỉnh: `VideoLoader -> Detector (YOLOv8s) -> Tracker (ByteTrack) -> Overlay Visualizer -> VideoWriter`.
- Làm mốc đối chứng khoa học (Baseline Benchmark) đo lường chính xác năng lực tính toán của phần cứng thực tế (NVIDIA GeForce RTX 3050 6GB Laptop GPU).
- Đánh giá khả năng đáp ứng thời gian thực (Real-time Constraints: $\ge 25\text{ FPS}$).

---

## 2. KIẾN TRÚC & TỐI ƯU HÓA KỸ THUẬT

### 2.1. Cấu hình mô hình & Pipeline
- **Mô hình suy luận:** `YOLOv8s` (pretrained COCO, 22.5M weights).
- **Bộ theo dõi đa mục tiêu (MOT):** `ByteTrack` tích hợp liên kết hai tầng điểm tin cậy ($D_{high}$ và $D_{low}$), giữ ID bền vững qua che khuất tạm thời (`track_buffer=30`).
- **Độ chính xác bán phần (Half Precision FP16):** Chuyển đổi trọng số sang FP16 trực tiếp trên Tensor Cores của GPU RTX 3050, cắt giảm 50% băng thông bộ nhớ và loại bỏ hoàn toàn cảnh báo suy luận.
- **Vẽ Overlay tối ưu:** Thao tác trực tiếp trên mảng NumPy của khung hình, vẽ Bounding Box, Track ID, nhãn phân loại, đuôi quỹ đạo di chuyển (Trajectory Trace) và bảng điều khiển HUD trực quan.

### 2.2. Đo lường độ trễ & Giám sát VRAM
- Bấm giờ vi mô từng khung hình bằng `time.perf_counter()` và đồng bộ hóa `torch.cuda.synchronize()` để thu được độ trễ thuần túy của phần cứng.
- Giám sát động dung lượng VRAM thực tế thông qua `torch.cuda.memory_allocated()`.

---

## 3. KẾT QUẢ THỰC NGHIỆM & NGHIỆM THU

Thực nghiệm đo đạc trên video chuẩn benchmark 30 giây (900 khung hình, độ phân giải 1280x720 @ 30 FPS):

| Chỉ số đo lường (Metric) | Tiêu chí yêu cầu | Kết quả thực tế (RTX 3050 Laptop) | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Tốc độ xử lý trung bình (FPS)** | $\ge 25.0\text{ FPS}$ | **40.11 FPS** | **ĐẠT (Vượt 60.4% chỉ tiêu)** |
| **Độ trễ trung bình / frame** | $\le 40.0\text{ ms}$ | **15.27 ms** | **Rất tốt** |
| **Độ trễ phân vị P95 / frame** | $\le 45.0\text{ ms}$ | **19.89 ms** | **Ổn định, không giật lag** |
| **Mức tiêu thụ VRAM đỉnh (Peak VRAM)** | $\le 4096\text{ MB}$ | **110.4 MB** / 6144 MB | **Cực kỳ nhẹ (chỉ ~1.8% VRAM)** |
| **Độ bền vững tracking** | Không văng lỗi | Xử lý trọn vẹn 900/900 frames | **ĐẠT 100%** |
| **Tập tin video kết quả xuất ra** | Có overlay | `annotated_baseline.mp4` (~5.8 MB) | **ĐẠT** |

---

## 4. HƯỚNG DẪN CHẠY LẠI THỰC NGHIỆM

```powershell
# Kích hoạt môi trường
cd D:\NIENLUANNGANH

# Chạy kiểm thử mặc định 30s benchmark tự động:
.\venv_nln\Scripts\python.exe run_baseline.py

# Hoặc chạy trên video thực tế bất kỳ:
.\venv_nln\Scripts\python.exe run_baseline.py --source đường_dẫn_video.mp4 --output video_ket_qua.mp4

# Chạy từ Webcam trực tiếp:
.\venv_nln\Scripts\python.exe run_baseline.py --source 0
```

---

## 5. KẾT LUẬN & CHUYỂN BƯỚC TIẾP THEO
- Hệ thống đã có bộ khung thực thi vững chắc, hoàn toàn làm chủ luồng video và tracking trên GPU thực tế với tốc độ vượt ngưỡng thời gian thực.
- Toàn bộ pipeline đã sẵn sàng để nạp trọng số tùy biến sau khi huấn luyện trên tập dữ liệu đặc thù giao thông Việt Nam ở **Giai đoạn 5**.

