# Chiến lược Dữ liệu & Thiết kế Thực nghiệm Khoa học
## Dự án Niên Luận — Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
**Sinh viên:** Nguyễn Thanh Bình — B2306518 — CT466E — ĐH Cần Thơ  
**GVHD:** TS. Lâm Nhựt Khang  
**Ngày:** 2026-11-10

---

## 1. Quy Trình Tổ Chức Dữ Liệu (Video-level Split)

### 1.1 Lý do phải dùng Video-level Split

> [!IMPORTANT]
> **Không được chia frame ngẫu nhiên (frame-level random split)!**
> 
> Các frame liên tiếp trong một video có nội dung gần giống nhau. Nếu frame F₁₀₀ vào **Train** và frame F₁₀₁ vào **Test**, mô hình sẽ "nhìn thấy" nội dung test trước khi đánh giá → **Data Leakage → kết quả mAP ảo, không phản ánh thực tế**.

**Giải pháp: Video-level Split**
- Mỗi video được gán **nguyên vẹn** vào đúng 1 split.
- Toàn bộ frame của video đó chỉ nằm trong 1 split duy nhất.
- Kiểm tra chéo bằng `data_checker.py` để xác nhận không có filename trùng.

### 1.2 Quy trình 5 bước

```
Bước 1: Thu thập video thô
        └─ Camera giao thông thực tế, ≥ 1080p, MP4/AVI
        └─ Mục tiêu: 5–10 video × 5–10 phút mỗi video

Bước 2: Trích frame
        └─ python scripts/organize_dataset.py extract --fps 2
        └─ 2 FPS × 5 phút = 600 frame/video → đủ đa dạng, không dư thừa

Bước 3: Gán nhãn (Annotation)
        └─ Công cụ: Roboflow (online) hoặc CVAT (local) hoặc LabelImg
        └─ Format: YOLO TXT (x_center y_center width height, normalized)
        └─ 5 lớp: car, motorcycle, bus, truck, traffic_light

Bước 4: Video-level Split
        └─ python scripts/organize_dataset.py split --ratio 0.70 0.15 0.15
        └─ Seed cố định 42 để tái tạo thực nghiệm

Bước 5: Kiểm tra toàn vẹn
        └─ python scripts/data_checker.py --verbose
        └─ Phải đạt: 0 lỗi, 0 data leakage
```

### 1.3 Cấu trúc thư mục dataset

```
D:\NIENLUANNGANH\dataset\
├── data.yaml                   ← Cấu hình YOLO (5 lớp, đường dẫn split)
├── _frames_scratch\            ← Vùng trung gian (xóa sau khi split xong)
│   ├── video_01\
│   │   ├── video_01_frame000000.jpg
│   │   ├── video_01_frame000000.txt   ← nhãn tương ứng
│   │   └── ...
│   └── video_02\ ...
├── images\
│   ├── train\                  ← ~70% số video → toàn bộ frame của chúng
│   ├── val\                    ← ~15% số video
│   └── test\                   ← ~15% số video (giữ kín cho đánh giá cuối)
└── labels\
    ├── train\
    ├── val\
    └── test\
```

---

## 2. Thiết Kế file data.yaml

**File:** [`D:\NIENLUANNGANH\dataset\data.yaml`](file:///D:/NIENLUANNGANH/dataset/data.yaml)

| Lớp | ID | Mô tả | Annotation rule |
|---|---|---|---|
| `car` | 0 | Ô tô con, SUV, minivan | BBox bao toàn thân + gương |
| `motorcycle` | 1 | Xe máy, xe gắn máy | BBox bao người + xe + mũ bảo hiểm |
| `bus` | 2 | Xe buýt, xe khách | BBox toàn thân kể cả nóc xe |
| `truck` | 3 | Xe tải các loại | BBox toàn thân kể cả thùng hàng |
| `traffic_light` | 4 | Đèn tín hiệu | Chỉ bao phần bóng đèn, không kể cột |

> [!NOTE]
> `motorcycle` (ID=1) là lớp quan trọng nhất và khó nhất. Xe máy Việt Nam đa dạng kiểu dáng, chen lấn, bị che khuất. Cần đảm bảo lớp này chiếm ít nhất **40–50% tổng annotations** (tương ứng thực tế giao thông VN).

---

## 3. Chiến Lược Data Augmentation cho Camera Giao Thông Ngoài Trời

### 3.1 Cơ sở lựa chọn

Camera giao thông ngoài trời chịu ảnh hưởng từ:

| Yếu tố môi trường | Tác động | Augmentation bù đắp |
|---|---|---|
| Nắng chói buổi sáng/chiều | Overexposure vùng sáng | Brightness ↑, Gamma correction |
| Mưa/sương mù | Giảm độ tương phản, blur | Blur, Fog simulation |
| Bóng đổ | Vùng tối cục bộ | Shadow augmentation |
| Ban đêm | Ánh đèn xe, Low-light | Brightness ↓↓, Noise |
| Góc camera cố định | Ít biến đổi phối cảnh | Perspective transform nhẹ |
| Mật độ xe cao | Xe bị che khuất (occlusion) | Mosaic 4-ảnh, Copy-paste |

### 3.2 Cấu hình Augmentation trong YOLO (config.yaml)

Ultralytics YOLO hỗ trợ augmentation trực tiếp qua tham số training:

```yaml
# Augmentation parameters — tối ưu cho camera giao thông ngoài trời
augment: true

# ── Màu sắc & ánh sáng (quan trọng nhất với outdoor camera) ──
hsv_h: 0.015      # ± hue shift 1.5% → bù biến đổi màu đèn giao thông
hsv_s: 0.700      # ± saturation 70% → bù mưa, sương, ánh sáng yếu
hsv_v: 0.400      # ± value/brightness 40% → bù nắng chói / đêm tối

# ── Hình học (nhẹ — camera giao thông ít thay đổi góc) ──
degrees:  0.0     # Không xoay (camera cố định, không có cảnh nghiêng)
translate: 0.1    # Dịch chuyển ảnh ±10%
scale:    0.5     # Zoom in/out ±50% → bù xe gần/xa
shear:    0.0     # Không shear (camera overhead/side cố định)
perspective: 0.0001  # Perspective transform rất nhẹ

# ── Lật ──
flipud:  0.0      # Không lật dọc (cảnh giao thông có hướng trên/dưới)
fliplr:  0.5      # Lật ngang 50% → bù camera nhìn từ 2 phía đường

# ── Ghép ảnh — quan trọng để học occlusion ──
mosaic:  1.0      # Luôn bật Mosaic 4-ảnh → tăng mật độ đối tượng/frame
mixup:   0.1      # Mixup 10% → tổng quát hóa
copy_paste: 0.1   # Copy-paste 10% → tăng xe máy bị che khuất

# ── Erasing ──
erasing: 0.4      # Random erasing 40% → bù occlusion thực tế
```

### 3.3 Augmentation thủ công bổ sung (nếu dataset nhỏ)

Nếu tổng dataset < 2000 ảnh, áp dụng thêm augmentation offline bằng Albumentations:

```python
import albumentations as A

outdoor_traffic_transform = A.Compose([
    # Mô phỏng mưa & sương mù
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.10),

    # Mô phỏng ánh sáng thực tế
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),

    # Mô phỏng bóng đổ
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.2),

    # Nhiễu & blur (camera ngoài trời đôi khi bị rung)
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.MotionBlur(blur_limit=5, p=0.15),

    # Nén JPEG (camera IP thường nén mạnh)
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

> [!TIP]
> **Thứ tự ưu tiên augmentation cho dự án này:**
> 1. `hsv_v` (brightness) — quan trọng nhất với outdoor lighting
> 2. `mosaic` — quan trọng nhất với occlusion xe máy
> 3. `fliplr` — miễn phí, hiệu quả cao
> 4. `RandomFog/Rain` — thực tế Việt Nam có mưa nhiều

---

## 4. Tiêu Chuẩn Đánh Giá Thực Nghiệm

### 4.1 Object Detection — đánh giá bằng mAP

**mAP@0.5** (IoU threshold = 0.5):
$$\text{mAP@0.5} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c(\text{IoU} \geq 0.5)$$

**mAP@0.5:0.95** (trung bình qua 10 IoU threshold từ 0.5 đến 0.95):
$$\text{mAP@0.5:0.95} = \frac{1}{10} \sum_{t \in \{0.5, 0.55, ..., 0.95\}} \text{mAP}(t)$$

| Chỉ tiêu | Mức tối thiểu | Mức tốt | Mức xuất sắc |
|---|---|---|---|
| mAP@0.5 | ≥ 0.70 | ≥ 0.80 | ≥ 0.90 |
| mAP@0.5:0.95 | ≥ 0.45 | ≥ 0.55 | ≥ 0.65 |
| Precision@0.5 | ≥ 0.75 | ≥ 0.85 | ≥ 0.90 |
| Recall@0.5 | ≥ 0.70 | ≥ 0.80 | ≥ 0.88 |

**Lưu ý:** Báo cáo kết quả **per-class** để phân tích lớp `motorcycle` riêng biệt.

### 4.2 Multi-Object Tracking — đánh giá bằng HOTA & IDF1

**HOTA (Higher Order Tracking Accuracy)** — Luiten et al., IJCV 2021:
$$\text{HOTA} = \sqrt{\text{DetA} \times \text{AssA}}$$
- DetA: Detection Accuracy (phát hiện đúng bao nhiêu)
- AssA: Association Accuracy (theo dõi đúng qua thời gian)

**IDF1 (Identification F1)**:
$$\text{IDF1} = \frac{2 \cdot \text{IDP} \times \text{IDR}}{\text{IDP} + \text{IDR}}$$

| Chỉ tiêu | Mức tối thiểu | Mức tốt | Ghi chú |
|---|---|---|---|
| HOTA | ≥ 0.55 | ≥ 0.65 | Cân bằng detect + assoc |
| IDF1 | ≥ 0.65 | ≥ 0.75 | Nhận diện ID nhất quán |
| ID Switches | ≤ 150/video | ≤ 80/video | Đổi ID → vi phạm bỏ sót |
| MOTA | ≥ 0.60 | ≥ 0.70 | Tracking tổng thể |
| FP Rate | ≤ 10% | ≤ 5% | Báo động giả |

### 4.3 Phát hiện vi phạm — đánh giá Event-level F1

> [!IMPORTANT]
> **Không dùng frame-level metric cho vi phạm** — một sự kiện vi phạm kéo dài nhiều frame. Phải dùng **Event-level** (1 vi phạm = 1 sự kiện, không tính frame).

**Định nghĩa:**
- **True Positive (TP):** Sự kiện vi phạm thực tế được phát hiện đúng (IoU thời gian ≥ 0.5)
- **False Positive (FP):** Báo vi phạm nhưng thực tế không vi phạm
- **False Negative (FN):** Vi phạm thực tế nhưng bỏ sót

$$\text{Precision}_{\text{event}} = \frac{TP}{TP + FP}, \quad \text{Recall}_{\text{event}} = \frac{TP}{TP + FN}$$
$$\text{F1}_{\text{event}} = \frac{2 \cdot P \cdot R}{P + R}$$

| Chỉ tiêu | Mức tối thiểu | Mức tốt |
|---|---|---|
| Event Precision | ≥ 0.80 | ≥ 0.90 |
| Event Recall | ≥ 0.75 | ≥ 0.85 |
| Event F1 | ≥ 0.78 | ≥ 0.87 |

### 4.4 Hệ thống tổng thể — FPS

| Chỉ tiêu | Mức tối thiểu | Lý do |
|---|---|---|
| FPS (video 1080p) | ≥ 20 FPS | Xử lý real-time (25 FPS camera) |
| FPS (video 720p) | ≥ 30 FPS | Hiển thị mượt |
| Thời gian khởi động | ≤ 10s | Load model YOLO11s |
| VRAM sử dụng | ≤ 5.5GB | Giữ headroom 0.5GB trên RTX 3050 6GB |

---

## 5. Kế Hoạch Thu Thập Dữ Liệu

### 5.1 Yêu cầu dữ liệu tối thiểu

| Thành phần | Số lượng tối thiểu | Mục tiêu |
|---|---|---|
| Số video | 5 video | 8–10 video |
| Độ dài mỗi video | ≥ 5 phút | 5–15 phút |
| Độ phân giải | ≥ 720p | 1080p |
| Tổng frame sau trích | ≥ 2000 frame | 5000–8000 frame |
| Tổng annotation | ≥ 10000 bbox | 25000–40000 bbox |

### 5.2 Phân bố lớp mục tiêu (đặc thù giao thông Việt Nam)

```
motorcycle    : 45–55%  (xe máy chiếm đa số tại VN)
car           : 25–35%  (ô tô con)
truck         : 5–10%   (xe tải)
bus           : 3–7%    (xe buýt)
traffic_light : 5–10%   (cần đủ mẫu cho cả 3 trạng thái đỏ/vàng/xanh)
```

### 5.3 Nguồn dữ liệu khuyến nghị

1. **Tự quay:** Dùng điện thoại cố định tại nút giao Cần Thơ (ưu tiên — dữ liệu thực tế nhất)
2. **Camera CCTV công cộng:** Liên hệ sở GTVT Cần Thơ
3. **Dataset mở:** UA-DETRAC (xe hơi, Trung Quốc), VeRA Dataset
4. **Tổng hợp:** Kết hợp dữ liệu tự quay + public dataset để đủ số lượng

> [!WARNING]
> Dataset COCO chứa ít xe máy kiểu Việt Nam. Mô hình pre-trained trên COCO có thể nhận dạng motorcycle kém với xe số/xe điện 3 bánh VN. Cần **fine-tune** với ít nhất 300–500 ảnh xe máy Việt Nam.

---

## 6. Script Hỗ Trợ

| Script | Chức năng | Lệnh chạy |
|---|---|---|
| [`organize_dataset.py`](file:///D:/NIENLUANNGANH/scripts/organize_dataset.py) | Trích frame + Video-level Split | `python scripts/organize_dataset.py extract` |
| [`data_checker.py`](file:///D:/NIENLUANNGANH/scripts/data_checker.py) | Kiểm tra toàn vẹn nhãn + Leakage | `python scripts/data_checker.py --verbose` |

---

## Tài Liệu Tham Khảo

- Luiten, J. et al. (2021). HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. *IJCV*.
- Everingham, M. et al. (2010). The Pascal VOC Challenge. *IJCV*, 88(2), 303–338.
- Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV*.
- Buslaev, A. et al. (2020). Albumentations: Fast and Flexible Image Augmentations. *Information*, 11(2), 125.

