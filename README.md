# He Thong Tu Dong Phat Hien Phuong Tien Vi Pham Den Tin Hieu Giao Thong

> **De tai Nien Luan** | Nguyen Thanh Binh - CT466E - B2306518  
> **Truong Dai hoc Can Tho** | 2025-2026

## Phan cung yeu cau
| Thanh phan | Thong so |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop |
| CPU | Intel Core i5-12450HX (12 luong) |
| RAM | 12 GB DDR4 |
| OS  | Windows 11 Home 64-bit |
| CUDA | 12.4+ (Driver 581.86) |
| Python | 3.11.9 |

## Cau truc thu muc

```
D:\NIENLUANNGANH\
├── src/                      # Ma nguon chinh
│   ├── video_loader.py       # Doc/chuan hoa video
│   ├── detector.py           # YOLO wrapper
│   ├── tracker.py            # ByteTrack / BoT-SORT interface
│   ├── traffic_light.py      # Nhan dien trang thai den
│   ├── violation_logic.py    # Logic phat hien vi pham
│   └── evidence.py           # Trich xuat bang chung
├── dataset/                  # Du lieu huan luyen
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── models/
│   ├── pretrained/           # Model YOLO goc
│   └── checkpoints/          # Model sau khi fine-tune
├── videos/
│   ├── raw/                  # Video dau vao thu
│   └── annotated/            # Video ket qua
├── evidence/                 # Bang chung vi pham duoc ghi lai
├── experiments/              # Ket qua training YOLO
├── logs/                     # Log he thong
├── reports/figures/          # Bieu do, do thi bao cao
├── scripts/                  # Script tien ich
├── config.yaml               # Cau hinh trung tam
├── requirements.txt          # Cac thu vien Python
├── test_env.py               # Kiem tra moi truong
└── setup_env.bat             # Script cai dat tu dong (Windows)
```

## Cai dat nhanh

```powershell
# Buoc 1: Tao va kich hoat moi truong ao
python -m venv D:\NIENLUANNGANH\venv_nln
D:\NIENLUANNGANH\venv_nln\Scripts\activate

# Buoc 2: Cai dat PyTorch CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Buoc 3: Cai dat cac thu vien con lai
pip install -r requirements.txt

# Buoc 4: Kiem tra moi truong
python test_env.py
```

Hoac chay toan bo tu dong: `setup_env.bat`

## Cac giai doan thuc hien

| Giai doan | Noi dung | Trang thai |
|---|---|---|
| 0 | Thiet lap moi truong | ✅ Hoan thanh |
| 1 | Nghien cuu tong quan | ⬜ Chua bat dau |
| 2 | Du lieu & thiet ke thuc nghiem | ⬜ Chua bat dau |
| 3 | Thiet ke kien truc pipeline | ⬜ Chua bat dau |
| 4 | Xay dung Baseline | ⬜ Chua bat dau |
| 5 | Huan luyen & toi uu mo hinh | ⬜ Chua bat dau |
| 6 | Logic vi pham & xu ly ca kho | ⬜ Chua bat dau |
| 7 | Hoan thien giao dien Demo | ⬜ Chua bat dau |
| 8 | Danh gia tong the & bao cao | ⬜ Chua bat dau |

## Tieu chi khoa hoc

- **Detection**: mAP@0.5 >= 0.80, mAP@0.5:0.95 >= 0.55
- **Tracking**: IDF1 >= 0.70, HOTA >= 0.60
- **Vi pham**: Event Precision >= 0.85, Event Recall >= 0.75
- **Hieu nang**: >= 30 FPS tren RTX 3050 6GB (inference realtime)

