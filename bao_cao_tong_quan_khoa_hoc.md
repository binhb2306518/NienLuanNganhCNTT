# BÁO CÁO TỔNG QUAN KHOA HỌC — GIAI ĐOẠN 1
## Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
### Sinh viên: Nguyễn Thanh Bình — B2306518 — CT466E
### GIẢNG VIÊN HD: TS. Lâm Nhựt Khang | Đại học Cần Thơ, 11/2026

---

## 1. BỐI CẢNH NGHIÊN CỨU VÀ CÂU HỎI KHOA HỌC

### 1.1. Tính cấp thiết

Hành vi vượt đèn đỏ là một trong những nguyên nhân gây tai nạn giao thông hàng đầu tại Việt Nam. Theo Ủy ban An toàn giao thông Quốc gia, hơn 30% số vụ tai nạn tại các nút giao có đèn tín hiệu liên quan đến vi phạm hiệu lệnh. Các hệ thống giám sát thủ công hiện tại (camera CCTV với nhân viên trực) chỉ có khả năng xử phạt một phần nhỏ trong tổng số vi phạm, thiếu tính nhất quán và không thể hoạt động liên tục 24/7.

Những hệ thống phát hiện vi phạm tự động đã được triển khai tại các quốc gia phát triển nhưng **chưa được nghiên cứu và tối ưu hóa đặc thù cho điều kiện giao thông hỗn hợp Việt Nam**, nơi mật độ xe máy chiếm trên 70% tổng phương tiện, với hành vi di chuyển khó dự đoán và mức độ che khuất lẫn nhau cao.

### 1.2. Câu hỏi nghiên cứu (Research Questions)

**RQ1 — Detection:** Với phần cứng hạn chế (RTX 3050 6GB), kiến trúc YOLO nào (YOLOv8s vs. YOLO11s) đạt được sự cân bằng tốt nhất giữa mAP@0.5:0.95 và tốc độ suy luận (FPS) trên luồng video giao thông hỗn hợp Việt Nam?

**RQ2 — Tracking:** Trong điều kiện camera ngoài trời với rung lắc cơ học và mật độ xe máy cao (tỷ lệ che khuất tạm thời lớn), thuật toán nào giữa ByteTrack và BoT-SORT cho chỉ số IDF1 và số lần hoán đổi định danh (ID Switches) tốt hơn?

**RQ3 — Traffic Light:** Pipeline phân loại trạng thái đèn tín hiệu nào đạt độ chính xác tốt hơn tại Việt Nam (phát hiện trực tiếp YOLO vs. phân tích màu HSV hai bước), đặc biệt trong điều kiện thời tiết thay đổi và ánh sáng ngược?

**RQ4 — Violation Logic:** Cơ chế cảm biến hai vạch ảo (Dual Virtual Line) có giúp giảm tỷ lệ False Positive so với phương pháp một vạch truyền thống không, và bộ lọc thời gian trễ đèn đỏ ảnh hưởng như thế nào đến tỷ lệ False Negative?

**RQ5 — System:** Toàn bộ pipeline (YOLO + Tracker + Violation Logic) đạt bao nhiêu FPS trên RTX 3050 6GB, và có đáp ứng được ngưỡng thời gian thực ≥ 25 FPS không?

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Kiến trúc YOLO: So sánh YOLOv8 và YOLO11

#### 2.1.1. Tổng quan họ mô hình YOLO

YOLO (You Only Look Once) là họ mô hình phát hiện đối tượng **một giai đoạn** (single-stage), xử lý toàn bộ ảnh trong một lần suy luận duy nhất thay vì đề xuất vùng quan tâm như các kiến trúc hai giai đoạn (Faster R-CNN). Điều này tạo ra tốc độ xử lý vượt trội, phù hợp cho bài toán thời gian thực.

#### 2.1.2. YOLOv8 (Ultralytics, 2023)

YOLOv8 được phát hành bởi Ultralytics vào tháng 1/2023, kế thừa từ YOLOv5 với nhiều cải tiến kiến trúc quan trọng:

**Backbone — CSPDarknet với C2f Block:**
- Block **C2f** (Cross Stage Partial with 2 convolution and feature fusion) thay thế C3 của YOLOv5
- Ghép nối tất cả các đầu ra từ Bottleneck layers, tạo gradient dày đặc hơn
- Cấu trúc: `Conv → Split → N×Bottleneck → Concat → Conv`

**Neck — FPN + PAN (Feature Pyramid Network + Path Aggregation Network):**
- Kết hợp top-down (FPN) và bottom-up (PAN) để truyền đặc trưng đa tỷ lệ
- Ba đầu ra tương ứng ba mức phân giải: P3 (80×80), P4 (40×40), P5 (20×20)

**Head — Anchor-free, Decoupled:**
- **Không dùng anchor boxes** (anchor-free): dự đoán trực tiếp khoảng cách từ điểm lưới tới 4 cạnh hộp bao
- **Decoupled head**: tách biệt nhánh phân loại (classification) và nhánh hồi quy tọa độ (regression) — cải thiện hội tụ
- **Task-Aligned Assigner** (Feng et al., 2021) gán nhãn trong huấn luyện dựa trên tích giữa điểm phân loại và IoU

**Hàm mất mát:**
- Hồi quy: **CIoU Loss** (Complete IoU) = 1 - IoU + ρ²(b, bgt)/c² + αv
- Phân phối tọa độ: **DFL** (Distribution Focal Loss) — mô hình hóa vị trí cạnh như phân phối xác suất rời rạc
- Phân loại: **Binary Cross-Entropy** với nhãn mềm từ Task-Aligned Assigner

#### 2.1.3. YOLO11 (Ultralytics, 2024)

YOLO11 được phát hành vào tháng 9/2024, giữ lại triết lý thiết kế anchor-free của YOLOv8 nhưng thay thế các khối trích xuất đặc trưng cốt lõi:

**Backbone — C3k2 Block (thay thế C2f):**
```
Input → Conv1×1 → [C3k2 Block] → Conv1×1 → Output
C3k2 = Cross Stage Partial với 2 convolution kernel kích thước k (mặc định k=2)
```
- Cho phép **kernel nhỏ hơn** so với C2f, giảm FLOPs trong khi duy trì độ sâu ngữ nghĩa
- Hiệu quả hơn trên vật thể nhỏ nhờ receptive field được điều chỉnh

**Neck — C2PSA (Cross Stage Partial with Position-Sensitive Attention):**
- **PSA Module**: thêm cơ chế chú ý vị trí (Positional Self-Attention) vào C2f
- Công thức: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Mỗi đặc trưng pixel "nhìn" vào tất cả vị trí khác trong feature map, cải thiện khả năng phát hiện vật thể bị che khuất một phần
- **Hiệu quả với xe máy** bị che khuất bởi xe buýt hoặc xe tải trong luồng giao thông Việt Nam

**So sánh kiến trúc YOLOv8s vs. YOLO11s:**

| Tiêu chí | YOLOv8s | YOLO11s |
|---|---|---|
| Backbone Block | C2f | C3k2 |
| Neck Attention | Không có | C2PSA (Position-Sensitive) |
| Head | Anchor-free, Decoupled | Anchor-free, Decoupled |
| Hàm mất mát | CIoU + DFL + BCE | CIoU + DFL + BCE |
| Tham số | ~11.2M | ~9.4M (nhẹ hơn 16%) |
| mAP@0.5:0.95 (COCO) | 44.9% | 47.0% (tốt hơn 2.1%) |
| Tốc độ GPU (T4 TensorRT) | 1.47 ms | 1.27 ms |
| **Điểm mạnh đặc thù** | Ổn định, cộng đồng lớn | Vật thể nhỏ, giảm ID switch |

> **Lựa chọn của dự án:** YOLO11s là mô hình chính; YOLOv8s là baseline đối chứng. Khối C2PSA trong YOLO11s dự kiến giúp phân biệt xe máy che khuất nhau tốt hơn trong điều kiện giao thông hỗn hợp.

---

### 2.2. Thuật toán Theo dõi Đa Đối tượng (Multi-Object Tracking)

#### 2.2.1. Tổng quan hướng tiếp cận Tracking-by-Detection

Hướng tiếp cận **Tracking-by-Detection** (Bewley et al., 2016) chia bài toán MOT thành hai bước tách biệt:
1. **Detection**: Mô hình YOLO phát hiện các đối tượng trong từng khung hình độc lập
2. **Association**: Thuật toán tracker liên kết các phát hiện theo thời gian, duy trì Track ID nhất quán

Thành phần cốt lõi:
- **Bộ lọc Kalman**: Ước lượng và dự đoán trạng thái chuyển động (vị trí, vận tốc)
- **Thuật toán Hungarian** (hay LAPJV): Giải bài toán gán liên kết tối ưu trên ma trận chi phí IoU

#### 2.2.2. ByteTrack (Zhang et al., ECCV 2022)

**Đóng góp chính**: Tận dụng phát hiện có điểm tin cậy THẤP thay vì loại bỏ chúng.

**Cơ chế liên kết hai giai đoạn:**

```
Phát hiện D_t được chia thành:
  D_high: confidence ≥ θ_high (mặc định 0.6)
  D_low:  θ_low ≤ confidence < θ_high (mặc định 0.1–0.6)

Giai đoạn 1: Tất cả track T ↔ D_high (Hungarian + IoU distance)
  → T_matched, T_remain, D_high_remain

Giai đoạn 2: T_remain ↔ D_low (chỉ dùng IoU)
  → Phục hồi xe máy bị che khuất tạm thời
  → D_low không ghép được → loại bỏ như nền

Quản lý vòng đời:
  - T_remain sau GĐ2 → trạng thái "Lost", giữ T_lost=30 frame
  - D_high_remain → khởi tạo track mới (xác nhận sau 1 frame kế)
```

**Ưu điểm cho bài toán:** Xe máy bị che khuất bởi xe buýt thường vẫn có hộp bao với confidence ~0.15–0.4; ByteTrack không loại bỏ chúng mà dùng để duy trì quỹ đạo, giảm ID Switches đáng kể.

**Giới hạn:** Không xử lý được camera rung lắc (chuyển động toàn cục của background).

#### 2.2.3. BoT-SORT (Aharon et al., 2022)

**Đóng góp chính**: Thêm Camera Motion Compensation (CMC) và tùy chọn ReID vào ByteTrack.

**Ba cải tiến cốt lõi:**

**[1] Điều chỉnh vectơ trạng thái Kalman:**
```
ByteTrack: x = [u, v, γ, h, u̇, v̇, γ̇, ḣ]  (γ = w/h tỷ lệ khung hình)
BoT-SORT:  x = [u, v, w, h, u̇, v̇, ẇ, ḣ]  (w, h chiều rộng và chiều cao)
```
Dùng trực tiếp w, h thay vì γ giúp dự đoán ổn định hơn khi xe thay đổi hướng.

**[2] Bù chuyển động camera (CMC) — quan trọng nhất:**
```
Các bước:
1. Trích xuất điểm đặc trưng nền bằng Shi-Tomasi corner detector
2. Theo dõi sang khung hình kế bằng Lucas-Kanade Optical Flow (thưa)
3. Ước lượng biến đổi Affine bằng RANSAC (loại bỏ điểm trên phương tiện)
4. Hiệu chỉnh trạng thái Kalman: x̂'_t = M̃·x̂_t + T̃

Lợi ích: Đưa vị trí dự đoán về cùng hệ quy chiếu với khung hình hiện tại
→ IoU giữa dự đoán và phát hiện không bị giảm do camera rung
→ Áp dụng được cho tọa độ vạch dừng và ROI cố định (quan trọng!)
```

**[3] Kết hợp đặc trưng chuyển động và ngoại hình (tùy chọn):**
```
Dùng EMA để cập nhật đặc trưng: e_i^t = β·e_i^(t-1) + (1-β)·f_i^t
Chi phí tổng hợp: C_ij = min(d_ij^IoU, d̂_ij^cos)
Chỉ kết hợp khi cả IoU và cosin distance đều dưới ngưỡng → tránh ghép sai
```

**Bảng so sánh chi tiết:**

| Tiêu chí | ByteTrack | BoT-SORT (CMC, không ReID) |
|---|---|---|
| Trạng thái Kalman | 8D (γ, h) | 8D (w, h) |
| Phát hiện điểm thấp | Có (giai đoạn 2) | Có (giai đoạn 2) |
| Bù camera rung | Không | **Có (Affine + RANSAC)** |
| Đặc trưng ReID | Không | Tùy chọn (tắt mặc định) |
| Chi phí tính toán | ~0.5 ms/frame | ~1.5–2 ms/frame (CMC) |
| Phù hợp nhất | Camera cố định | Camera ngoài trời, rung nhẹ |
| IDF1 (MOT17) | 77.3% | 78.7% |
| ID Switches (MOT17) | 1,223 | **1,019** (ít hơn 16.7%) |

> **Kế hoạch thực nghiệm:** Chạy song song ByteTrack và BoT-SORT trên cùng tập video, so sánh HOTA, IDF1, ID Switches và chi phí thời gian/frame. Lựa chọn theo điều kiện camera thực tế.

---

### 2.3. Phân loại Trạng thái Đèn Tín hiệu

#### 2.3.1. Hướng 1: Phát hiện trực tiếp bằng YOLO

Huấn luyện YOLO phân biệt 3 lớp: `traffic_light_red`, `traffic_light_yellow`, `traffic_light_green`.

**Ưu điểm:** End-to-end, không cần thiết kế thuật toán thủ công.

**Nhược điểm đặc thù Việt Nam:**
- Đèn giao thông Việt Nam có nhiều hình dạng và kích thước phi chuẩn
- Ánh sáng ngược (backlight) buổi sáng/chiều làm giảm chất lượng ảnh đèn
- Cần tập dữ liệu gán nhãn đèn Việt Nam chuyên biệt (không có sẵn)

#### 2.3.2. Hướng 2: Pipeline hai bước (ROI + HSV)

```
Bước 1: Xác định vùng đèn
  - Camera cố định → ROI tọa độ cố định (nhanh, ổn định)
  - Hoặc: YOLO detect "traffic_light" class → crop vùng đèn

Bước 2: Phân tích màu sắc trong không gian HSV
  Màu ĐỎ:    H∈[0°,10°]∪[170°,180°], S≥120, V≥70
  Màu VÀNG:  H∈[15°,35°], S≥120, V≥70
  Màu XANH:  H∈[45°,85°], S≥100, V≥70

  Quyết định: LightState = argmax(pixel_ratio_per_color)
  Ổn định: Chỉ xác nhận khi trạng thái duy trì ≥ T_stable frame (≈15)
```

**Ưu điểm:** Không cần dữ liệu gán nhãn đèn, nhanh (~0.5ms), dễ điều chỉnh ngưỡng.

**Nhược điểm:** Phụ thuộc vào điều kiện ánh sáng; cần điều chỉnh ngưỡng HSV theo từng địa điểm.

> **Phương án đề xuất của dự án:** Hướng 2 (ROI cố định + HSV) làm phương án chính với bộ lọc thời gian trễ ổn định. Hướng 1 làm thực nghiệm so sánh nếu có đủ dữ liệu gán nhãn.

---

### 2.4. Các Phương pháp Học máy Truyền thống Làm Baseline So sánh

Để trả lời câu hỏi phản biện "Tại sao cần Deep Learning?", đề tài thiết kế thực nghiệm so sánh trực tiếp trên bài toán phân loại trạng thái đèn tín hiệu:

**Đặc trưng thủ công (Hand-crafted Features):**
- **Color Histogram HSV**: Biểu đồ tần suất màu sắc (32 bins H, 8 bins S, 8 bins V = 384 chiều)
- **HOG** (Histogram of Oriented Gradients): Đặc trưng hình dạng cạnh (Dalal & Triggs, 2005)

**Các thuật toán ML đối chứng:**

| Thuật toán | Nguyên lý | Ưu điểm | Nhược điểm chính |
|---|---|---|---|
| K-Nearest Neighbors | Khoảng cách Euclidean, bỏ phiếu k láng giềng | Đơn giản, không huấn luyện | Chậm khi inference, nhạy cảm ánh sáng |
| Gaussian Naive Bayes | Bayes theorem + giả định độc lập | Rất nhanh (<1ms), ít bộ nhớ | Giả định độc lập không thực tế, kém khi ngược sáng |
| SVM (kernel RBF) | Siêu phẳng tối ưu biên trong không gian phi tuyến | Chính xác nhất trong ML cổ điển | Phụ thuộc chất lượng đặc trưng thủ công |
| Random Forest | Ensemble cây quyết định (Bootstrap + Feature Subsampling) | Kháng overfitting, diễn giải được | Không tự học phân bố không gian như CNN |
| **Deep Learning (CNN)** | End-to-end, tự học đặc trưng | Thích nghi mọi điều kiện ánh sáng | Cần GPU và dữ liệu nhiều hơn |

---

## 3. CÁC CÔNG TRÌNH KHOA HỌC LIÊN QUAN

### 3.1. Danh sách 10+ tài liệu tham khảo có trích dẫn đầy đủ

**[1] YOLOv8 — Mô hình phát hiện gốc (Baseline Detection)**
> Jocher, G., Chaurasia, A., & Qiu, J. (2023). **Ultralytics YOLOv8**. Ultralytics. https://github.com/ultralytics/ultralytics
>
> *Đóng góp chính:* Kiến trúc anchor-free với decoupled head và C2f block. Là nền tảng cho cả YOLOv8 và YOLO11. Mô hình baseline (YOLOv8s) của dự án.

**[2] YOLO11 — Mô hình phát hiện chính**
> Jocher, G., & Qiu, J. (2024). **Ultralytics YOLO11**. Ultralytics. https://github.com/ultralytics/ultralytics (released September 2024)
>
> *Đóng góp chính:* Khối C3k2 và C2PSA với cơ chế chú ý vị trí. Đạt mAP@0.5:0.95 = 47.0% trên COCO với ít tham số hơn YOLOv8s (9.4M so với 11.2M).

**[3] ByteTrack — Thuật toán theo dõi chính (Ứng viên 1)**
> Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**. *Proceedings of the European Conference on Computer Vision (ECCV 2022)*, Lecture Notes in Computer Science, 13682, 1–21. https://doi.org/10.1007/978-3-031-20047-2_1
>
> *Đóng góp chính:* Liên kết hai giai đoạn tận dụng phát hiện điểm tin cậy thấp. Đạt HOTA 63.1%, IDF1 77.3% trên MOT17. Tốc độ 30 FPS trên GPU V100.

**[4] BoT-SORT — Thuật toán theo dõi đối chứng (Ứng viên 2)**
> Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022). **BoT-SORT: Robust Associations Multi-Pedestrian Tracking**. *arXiv preprint arXiv:2206.14651*. https://arxiv.org/abs/2206.14651
>
> *Đóng góp chính:* Camera Motion Compensation (CMC) bằng Lucas-Kanade + RANSAC. Giảm ID Switches 16.7% so với ByteTrack trên MOT17. Phù hợp cho camera ngoài trời rung lắc.

**[5] SORT — Nền tảng lý thuyết Tracking-by-Detection**
> Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). **Simple Online and Realtime Tracking**. *2016 IEEE International Conference on Image Processing (ICIP)*, 3464–3468. https://doi.org/10.1109/ICIP.2016.7533003
>
> *Đóng góp chính:* Lần đầu đề xuất Tracking-by-Detection với Kalman Filter + Hungarian Algorithm. Nền tảng lý thuyết cho ByteTrack và BoT-SORT.

**[6] Task-Aligned Assigner — Phương pháp gán nhãn trong huấn luyện YOLO**
> Feng, C., Zhong, Y., Gao, Y., Scott, M. R., & Huang, W. (2021). **TOOD: Task-Aligned One-Stage Object Detection**. *Proceedings of the International Conference on Computer Vision (ICCV 2021)*, 3490–3499. https://doi.org/10.1109/ICCV48922.2021.00349
>
> *Đóng góp chính:* Bộ gán nhãn căn chỉnh theo nhiệm vụ (Task-Aligned Assigner), dùng trong cả YOLOv8 và YOLO11 để gán phát hiện vào ô lưới trong huấn luyện.

**[7] CIoU Loss — Hàm mất mát hồi quy tọa độ**
> Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). **Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression**. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(07), 12993–13000. https://doi.org/10.1609/aaai.v34i07.6999
>
> *Đóng góp chính:* CIoU Loss kết hợp IoU + khoảng cách tâm + tỷ lệ khung hình. Được dùng trong cả YOLOv8 và YOLO11 để huấn luyện nhánh hồi quy tọa độ.

**[8] DFL — Distribution Focal Loss cho tọa độ hộp bao**
> Li, X., Wang, W., Wu, L., Chen, S., Hu, X., Li, J., Tang, J., & Yang, J. (2020). **Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection**. *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 21002–21012.
>
> *Đóng góp chính:* Mô hình hóa vị trí cạnh hộp bao như phân phối xác suất rời rạc. Giúp xử lý đường biên mờ của phương tiện khi di chuyển nhanh.

**[9] Kalman Filter — Nền tảng toán học ước lượng chuyển động**
> Kalman, R. E. (1960). **A New Approach to Linear Filtering and Prediction Problems**. *Journal of Basic Engineering*, 82(1), 35–45. https://doi.org/10.1115/1.3662552
>
> *Đóng góp chính:* Thuật toán đệ quy tối ưu ước lượng trạng thái hệ động tuyến tính. Dùng để dự đoán vị trí phương tiện giữa các khung hình trong MOT.

**[10] HOG Features — Đặc trưng thủ công cho thực nghiệm đối chứng ML**
> Dalal, N., & Triggs, B. (2005). **Histograms of Oriented Gradients for Human Detection**. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2005)*, 886–893. https://doi.org/10.1109/CVPR.2005.177
>
> *Đóng góp chính:* Đặc trưng HOG (Histogram of Oriented Gradients) + SVM Sliding Window. Phương pháp truyền thống được dùng trong thực nghiệm đối chứng để so sánh với Deep Learning.

**[11] HOTA Metric — Chỉ số đánh giá tracking toàn diện**
> Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021). **HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking**. *International Journal of Computer Vision (IJCV)*, 129(2), 548–578. https://doi.org/10.1007/s11263-020-01375-2
>
> *Đóng góp chính:* HOTA (Higher Order Tracking Accuracy) — chỉ số toàn diện nhất cho MOT, cân bằng giữa Detection accuracy và Association accuracy. Được dùng làm chỉ số đánh giá chính trong dự án.

**[12] Lucas-Kanade Optical Flow — Nền tảng CMC trong BoT-SORT**
> Lucas, B. D., & Kanade, T. (1981). **An Iterative Image Registration Technique with an Application to Stereo Vision**. *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI 1981)*, 674–679.
>
> *Đóng góp chính:* Luồng quang học thưa Lucas-Kanade, dùng trong bước Camera Motion Compensation của BoT-SORT để ước lượng chuyển động toàn cục của camera.

**[13] Nghiên cứu giao thông Việt Nam — Bối cảnh địa phương**
> Trần Thị Thanh Thủy, & Nguyễn Hữu Nghĩa. (2021). **Ứng dụng thị giác máy tính trong giám sát giao thông tại Việt Nam: Thách thức và hướng tiếp cận**. *Tạp chí Khoa học Trường Đại học Cần Thơ*, 57(1A), 112–121.
>
> *Đóng góp chính:* Phân tích đặc thù giao thông hỗn hợp (xe máy chiếm 70%+) tại các nút giao đô thị Việt Nam, các thách thức về độ che khuất và điều kiện ánh sáng.

---

## 4. PHÂN TÍCH KHOẢNG TRỐNG NGHIÊN CỨU

### 4.1. Hạn chế của các công trình hiện có

**[Khoảng trống 1]: Thiếu dữ liệu và thực nghiệm đặc thù giao thông Việt Nam**
- Hầu hết nghiên cứu MOT dùng dataset MOT17/MOT20 (đi bộ ở châu Âu) hoặc BDD100K (giao thông Mỹ)
- Xe máy Việt Nam có đặc trưng độc đáo: nhỏ hơn xe máy phương Tây, thường đi theo nhóm, lách giữa xe to → tỷ lệ che khuất cao hơn nhiều
- **Đề xuất của dự án:** Thu thập và gán nhãn tập dữ liệu video giao thông Cần Thơ để fine-tune

**[Khoảng trống 2]: Chưa có phương pháp suy luận vi phạm tránh được False Positive cao**
- Nhiều nghiên cứu (Vu et al., 2020; Wei et al., 2022) chỉ dùng 1 vạch ảo → dẫn đến bắt nhầm xe dừng sát vạch hoặc xe rẽ phải hợp lệ
- **Đề xuất của dự án:** Cơ chế hai vạch ảo (Dual Virtual Line) + bộ lọc góc phương vị rẽ phải + bộ lọc thời gian trễ đèn đỏ (≥1 giây) → giảm FP

**[Khoảng trống 3]: Thiếu đánh giá Event-level (mức sự kiện vi phạm)**
- Các nghiên cứu thường chỉ báo cáo mAP (Detection) và IDF1/HOTA (Tracking)
- Chưa có chỉ số Event-level F1-score đánh giá chính xác hệ thống có bắt đúng vi phạm không
- **Đề xuất của dự án:** Xây dựng quy trình đánh giá Event-level (TP/FP/FN mức sự kiện)

**[Khoảng trống 4]: Chưa so sánh ByteTrack vs BoT-SORT trên điều kiện camera rung Việt Nam**
- Các bài báo so sánh tracker trên dataset chuẩn, không phản ánh điều kiện camera ngoài trời bị rung lắc do gió hoặc rung cơ học
- **Đề xuất của dự án:** Thực nghiệm đối chứng trực tiếp trên video giao thông Cần Thơ

**[Khoảng trống 5]: Thiếu thực nghiệm ML truyền thống vs Deep Learning trên bài toán phân loại đèn**
- Hầu hết nghiên cứu bỏ qua so sánh với ML cổ điển, thiếu luận giải khoa học về lý do chọn DL
- **Đề xuất của dự án:** Thực nghiệm đối chứng 4 thuật toán ML (KNN, Naive Bayes, SVM, RF) vs CNN trên tập ảnh crop đèn Việt Nam

---

## 5. KHUNG NGHIÊN CỨU ĐỀ XUẤT

```
                    ┌─────────────────────────────────────────────────┐
                    │          KHUNG NGHIÊN CỨU ĐỀ XUẤT               │
                    └─────────────────────────────────────────────────┘

[Dữ liệu]          Video giao thông nút giao Cần Thơ (1080p, 30fps)
    ↓               Thu thập thực tế + BDD100K + S2TLD làm bổ sung
    
[Phân chia]        Video-level split (không frame shuffle)
                    Train: 70% | Val: 15% | Test: 15% (nút giao chưa thấy)
    ↓
    
[Thực nghiệm 1]    Object Detection:  YOLOv11s vs YOLOv8s (Baseline)
  → Chỉ số:        mAP@0.5, mAP@0.5:0.95, FPS, VRAM
    ↓
    
[Thực nghiệm 2]    Multi-Object Tracking: ByteTrack vs BoT-SORT
  → Chỉ số:        HOTA, IDF1, MOTA, ID Switches, ms/frame
    ↓
    
[Thực nghiệm 3]    Traffic Light: HSV (ROI cố định) vs YOLO cls
  → Chỉ số:        Accuracy, F1, Latency ms
    ↓
    
[Thực nghiệm 4]    ML vs DL (đèn tín hiệu):
                    KNN, Naive Bayes, SVM, RF vs Mini-CNN
  → Chỉ số:        Accuracy, Macro F1, Confusion Matrix, ms/ảnh
    ↓
    
[Thực nghiệm 5]    Event-level Violation Detection (hệ thống tích hợp)
  → Chỉ số:        Event Precision, Event Recall, Event F1
    + Ablation Study: Vai trò bộ lọc thời gian, vai trò vạch ảo số 2
    ↓
    
[Kết quả]          Mô hình tốt nhất cho từng thành phần
                    + Phân tích lỗi (Error Analysis) + Giới hạn hệ thống
```

---

## 6. CẬP NHẬT CHECKLIST GIAI ĐOẠN 1

| Nhiệm vụ | Trạng thái |
|---|:---:|
| Tổng hợp lý thuyết YOLO (YOLOv8 vs YOLO11, C3k2, C2PSA) | ✅ Hoàn thành |
| Tổng hợp lý thuyết ByteTrack (liên kết 2 giai đoạn, D_low) | ✅ Hoàn thành |
| Tổng hợp lý thuyết BoT-SORT (CMC, Kalman vector điều chỉnh, ReID) | ✅ Hoàn thành |
| Trình bày ≥ 8 công trình khoa học có trích dẫn đầy đủ | ✅ Hoàn thành (13 tài liệu) |
| Phân tích khoảng trống nghiên cứu giao thông hỗn hợp Việt Nam | ✅ Hoàn thành (5 khoảng trống) |
| Xây dựng bộ câu hỏi nghiên cứu (RQ1–RQ5) | ✅ Hoàn thành |
| Thiết kế khung thực nghiệm 5 bước | ✅ Hoàn thành |
| Phân biệt kiến thức đã công bố và đề xuất của dự án | ✅ Hoàn thành |

