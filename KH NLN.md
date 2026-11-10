================================================================================
KE HOACH NIEN LUAN NGIEN CUU KHOA HOC VA TRIEN KHAI HE THONG AI
DE TAI: XAY DUNG HE THONG TU DONG PHAT HIEN PHUONG TIEN VI PHAM HIEU LENH DEN TIN HIEU GIAO THONG
================================================================================

Thong tin chung:
- Sinh vien thuc hien: Sinh vien Nganh Cong nghe Thong tin
- De tai: Xay dung he thong tu dong phat hien phuong tien vi pham hieu lenh den tin hieu giao thong
- Dinh huong: Nghien cuu sau ve AI, Thi giac may tinh, Mo hinh hoc sau, Thuc nghiem khoa hoc va Trien khai toi uu tren laptop
- Giao dien: Duy nhat mot giao dien demo tien loi, truc quan, chuyen nghiep
- File ke hoach: KH NLN.md (phien ban chuan hoa nghien cuu, cap nhat toan dien)

================================================================================
PHAN 1: DANH GIA HIEN TRANG TAI LIEU VA DINH HUONG LAI DE TAI
================================================================================

1.1. Danh gia cac tep tai lieu hien co trong thu muc

1. Tep mau_ke_hoach_nien_luan.md:
- Uu diem: Da phan chia 5 giai doan so khoi, nhan dien duoc cac module chinh (Detection, Traffic Light, Tracking, OCR).
- Thieu sot lon:
  + Mang tinh chat bieu mau de trong [...], chua co noi dung chi tiet va ke hoach hanh dong thuc te.
  + Thieu tieu chi danh gia khoa hoc chuan muc (mAP@0.5:0.95, IDF1, HOTA, Event-level F1).
  + Chua phan biet ro rang giua viec phat hien phuong tien va viec ket luan vi pham (thieu logic thoi gian va vach dung hop le).
  + Thieu cac mau prompt de sinh vien giao tiep voi AI theo tung buoc nghien cuu.

2. Tep nghien_cuu_cong_nghe.md:
- Uu diem: Khao sat phong phu cac mo hinh hien dai (YOLOv11, BoT-SORT, PaddleOCR).
- Sai lech nghiem trong can khac phuc ngay:
  + De xuat mot he thong Web Enterprise qua do so: FastAPI, Redis Queue, NestJS API Gateway, Socket.IO, MongoDB, MinIO, React 19, Tailwind CSS, Docker Compose, Nginx.
  + Sai lech dinh huong: Day la de tai Nien luan Nghien cuu AI cua sinh vien CNTT, khong phai do an phat trien phan mem web da tang.
  + Nguy co qua tai phan cung: Laptop sinh vien khong the ganh dong thoi Docker, co so du lieu phan tan va cac mo hinh Deep Learning chay song song ma van dam bao FPS thoi gian thuc.
  + Phan tan nguon luc: Sinh vien se mat 80% thoi gian de lap trinh web thay vi tap trung vao du lieu, huan luyen mo hinh, toi uu thuat toan va danh gia khoa hoc.

3. Tep chuong_1_gioi_thieu.md va tong_quan_de_tai.md:
- Da co phan tong quan tot ve tinh cap thiet, co so ly thuyet va cac nghien cuu lien quan tai DH Can Tho, DH Hang hai, the gioi (ByteTrack 2022).
- Can dong bo muc tieu thuc nghiem voi quy trinh khoa hoc chat che hon.

1.2. Dinh huong dieu chinh cot loi

- Chuyen trong tam 100% vao Nghien cuu AI va Thi giac may tinh.
- Huy bo kien truc microservices web phuc tap (NestJS, Redis, MinIO, React Web).
- Giu lai duy nhat 1 giao dien demo cuc bo (Desktop GUI su dung PyQt6 / PySide6 hoac Giao dien cuc bo nhe nhu Streamlit / Gradio) de trinh dien luong video, hien thi bounding box, trang thai den, quy dao xe, vach dung va hop thoai trich xuat bang chung vi pham.
- Toi uu toan bo pipeline chay muot tren laptop ca nhan voi GPU NVIDIA RTX 3050 6GB Laptop GPU.

================================================================================
PHAN 2: KIEM KE TAI NGUYEN PHAN CUNG VA MOI TRUONG LAPTOP THUC TE
================================================================================

2.1. Thong so he thong da kiem tra thuc te tren may

- He dieu hanh: Microsoft Windows 11 Home Single Language 64-bit
- Bo vi xu ly (CPU): 12th Gen Intel Core i5-12450HX (8 nhan vat ly, 12 luong, xung nhip co ban manh me)
- Bo nho trong (RAM): 12 GB (Hien dung khoang 10 GB, con trong khoang 1.9 GB can don dep tien trinh nen truoc khi train)
- Card do hoa (GPU): NVIDIA GeForce RTX 3050 6GB Laptop GPU
  + VRAM: 6144 MiB (6 GB GDDR6)
  + Driver Version: 581.86
  + Ho tro CUDA: CUDA 12.x / 13.0
- Dung luong o dia:
  + O dia C: Dung 215.1 GB, con trong 59.4 GB
  + O dia D: Dung 29.9 GB, con trong 170.3 GB (Noi luu tru ly tuong cho dataset, model checkpoint va video thuc nghiem)
- Moi truong lap trinh: Python 3.11.9 da cai dat san

2.2. Danh gia tinh kha thi va chien luoc su dung phan cung

1. Huan luyen (Training / Fine-tuning):
- VRAM 6GB hoan toan du de fine-tune cac phien ban mo hinh:
  + YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s (kich thuoc anh 640x640, batch-size tu 8 den 16, bat Mixed Precision FP16).
  + Neu thu nghiem YOLOv11m, can giam batch-size xuong 4 hoac 8 de tranh loi Out-Of-Memory (OOM).
- Chien luoc: Huan luyen tren o dia D de tan dung 170 GB dung luong trong.

2. Suy luan (Inference / Tracking):
- RTX 3050 6GB chay rat tot inference thoi gian thuc (30 - 60+ FPS) cho pipeline YOLO + Tracker khi su dung TensorRT hoac PyTorch FP16.
- Phuong an du phong CPU: Neu chay tren Intel i5-12450HX su dung ONNX Runtime hoac OpenVINO van dat tu 12 den 25 FPS voi cac model dang Nano/Small.

3. Quan ly RAM:
- Vi RAM vat ly la 12 GB, tuyet doi khong chay Docker Desktop, database nang hay cac ung dung ton bo nho trong luc train hoac chay pipeline AI.

================================================================================
PHAN 3: PHAN TICH BAI TOAN KHOA HOC VA PHUONG PHAP LUAN
================================================================================

3.1. Dinh nghia su kien vi pham mot cach khoa hoc

Mot sai lam pho bien la coi "khung hinh xuat hien phuong tien khi den do" la bang chung vi pham. Trong nghien cuu nay, he thong dinh nghia su kien vi pham dua tren chuoi logic khong gian - thoi gian (Spatio-Temporal Logic):

Dieu kien vi pham bat buoc hoi du tat ca cac yeu to:
1. Tin hieu den giao thong dang o trang thai DEN DO (Red Light Active) tai huong luu thong cua phuong tien, da duy tri on dinh it nhat qua T_buffer khung hinh (tranh nhap nhay).
2. Phuong tien co ID xuyen suot duoc gan boi bo theo doi (Tracker).
3. Vi tri phuong tien (diem tiep xuc banh xe hoac tam day bounding box) nam hoan toan phia truoc vach dung truoc thoi diem den do, hoac dang tien sat vach dung.
4. Phuong tien di chuyen vuot qua han mat phang vach dung (Stop Line) tien vao khu vuc giao lo trong khi den do van dang duy tri.
5. Van toc va huong di chuyen (Trajectory Vector) huong ve phia giao lo, loai tru truong hop xe lui lai.

3.2. Phan tich cac truong hop bien va ca kho (Edge Cases)

- Ca kho 1: Phuong tien can vach hoac dung sat vach roi dung han.
  Giai phap: Dat 2 vach ao (Virtual Lines): Vach 1 la Vach dung (Stop Line), Vach 2 la Vach xac nhan vi pham (Violation Confirmation Line cach vach dung khoang 1.5m - 2m). Chi khi xe vuot qua ca 2 vach moi ket luan vi pham.
- Ca kho 2: Phuong tien re phai tai noi co bien bao cho phep re phai khi den do.
  Giai phap: Thiet lap vung da giac (ROI) cho lan duoc phep re phai hoac loc huong vector di chuyen re phai.
- Ca kho 3: Che khuat (Occlusion) do mat do giao thong dong duc (dac trung xe may Viet Nam).
  Giai phap: Bo theo doi ByteTrack / BoT-SORT voi co che duy tri Track ID trong 30 khung hinh khi bi che khuat tam thoi.
- Ca kho 4: Den tin hieu o xa, do phan giai thap hoac bi anh sang chieu nguoc.
  Giai phap: Cho phep khoanh vung ROI co dinh cho den tin hieu hoac su dung bo phat hien chuyen biet crop vung den roi phan loai mau bang ket hop hoc sau va khong gian mau HSV.
- Ca kho 5: Camera bi rung lac do gio hoac rung co hoc.
  Giai phap: BoT-SORT co tinh nang Camera Motion Compensation (CMC) bang bien doi Affine giup triet tieu rung lac.

3.3. Pham vi nghien cuu ro rang

- Pham vi thuc hien:
  + Phat hien phuong tien: Xe may, O to con, Xe tai, Xe khach.
  + Phat hien den tin hieu: Do, Vang, Xanh.
  + Theo doi va bieu dien quy dao phuong tien.
  + Logic phat hien vuot den do va xuat bang chung: Video clip ngan 5-10 giay kem chuoi 3 anh (Truoc khi vuot, Thoi diem cat vach, Da vao giao lo).
- Pham vi mo rong (Uu tien so 2, neu du thoi gian):
  + Nhan dang bien so phuong tien bang PaddleOCR tren anh crop chat luong cao.

================================================================================
PHAN 4: KHAO SAT VA SO SANH CAC MO HINH UNG VIEN
================================================================================

4.1. Nhom mo hinh phat hien doi tuong (Object Detection)

Bang so sanh cac ung vien:
--------------------------------------------------------------------------------
Mo hinh     | Uu diem chinh             | Nhuoc diem             | Vi tri lua chon
--------------------------------------------------------------------------------
YOLOv8s     | On dinh, cong dong lon,    | Kien truc cu hon v11,  | Baseline doi chung
            | de trien khai va danh gia | do chinh xac thap hon  |
--------------------------------------------------------------------------------
YOLOv10s    | NMS-free, do tre suy luan | Kho fine-tune tren     | Phuong an du phong
            | rat thap                  | dataset tu tao         |
--------------------------------------------------------------------------------
YOLOv11s    | C3k2 + C2PSA tot cho vat  | Moi, can moi truong    | Ung vien chinh
            | nho, can bang speed/mAP   | Ultralytics moi nhat   | (Recommended)
--------------------------------------------------------------------------------
RT-DETR-l   | Chat luong cao, transformer| Nang, kho train tren   | Tham khao hoc thuat,
            | attention tot             | RTX 3050 6GB           | khong chon train
--------------------------------------------------------------------------------

Tieu chi lua chon mo hinh Detection:
- Mo hinh chinh duoc chon de phat trien: YOLOv11s (hoac YOLOv11n neu can toi da FPS).
- Mo hinh baseline doi chung: YOLOv8s.
- Tieu chi danh gia: mAP@0.5, mAP@0.5:0.95, FPS tren RTX 3050, dung luong VRAM chiem dung.

4.2. Nhom thuat toan theo doi doi tuong (Object Tracking)

Bang so sanh thuat toan tracking:
--------------------------------------------------------------------------------
Thuat toan  | Nguyen ly                  | Do phuc tap | Phu hop de tai
--------------------------------------------------------------------------------
ByteTrack   | Lien ket ca box diem cao  | Rat nhe,    | Ung vien sang gia 1
            | va box diem thap bang IOU | rat nhanh   | (Phu hop camera tinh)
--------------------------------------------------------------------------------
BoT-SORT    | Ket hop Kalman Filter,    | Trung binh, | Ung vien sang gia 2
            | ReID va bu chuyen dong CMC| on dinh cao | (Phu hop khi co rung lac)
--------------------------------------------------------------------------------
DeepSORT    | Su dung CNN trich xuat    | Nang, cham  | Loai bo vi da loi thoi,
            | dac trung ReID rieng      | hon         | de nhay ID
--------------------------------------------------------------------------------

Ke hoach thuc nghiem tracking:
- So sanh thuc nghiem giua ByteTrack va BoT-SORT tren cung mot tap video giao thong.
- Chi so danh gia: HOTA, IDF1, MOTA, so lan nhay ma dinh danh (ID Switches), do tre ms/frame.

4.3. Nhom phuong phap phan loai trang thai den tin hieu

Co 2 huong tiep can:
- Huong 1 (Phat hien truc tiep): Dung YOLO phat hien truc tiep cac lop traffic_light_red, traffic_light_yellow, traffic_light_green.
- Huong 2 (Pipeline 2 buoc): Buoc 1 khoanh vung hop den (bang YOLO hoac ROI co dinh cho camera co dinh). Buoc 2 su dung thuat toan phan tich mat do mau trong he khong gian mau HSV ket hop nguong sang.
Ke hoach: Thuc hien so sanh khoa hoc giua Huong 1 va Huong 2 tren tap du lieu video ngoai troi de chon huong co do on dinh cao nhat duoi su thay doi cua thoi tiet va goc chieu sang.

4.4. Thiet ke thuc nghiem doi chung: Hoc may truyen thong (Classical ML) va Hoc sau (Deep Learning)

Day la phan nghien cuu hoc thuat quan trong de dua vao Chuong 2 va Chuong 5 cua bao cao Nien luan, nham tra loi cau hoi phan bien: Vi sao can dung Deep Learning thay vi cac thuat toan Machine Learning co dien.

1. Bai toan thuc nghiem doi chung: Phan loai trang thai den tin hieu (Traffic Light State Classification: Do, Vang, Xanh, Tat).
- Dau vao: Tap anh crop ROI cua den tin hieu kich thuoc chuan hoa 32x32 hoac 64x64 pixel.
- Phuong phap trich xuat dac trung thu cong (Hand-crafted Features): Color Histogram trong khong gian mau HSV (32 bins cho H, 8 bins cho S, 8 bins cho V) ket hop dac trung canh HOG (Histogram of Oriented Gradients).

2. Danh sach cac giai thuat Machine Learning truyen thong duoc huan luyen va so sanh:
- Giai thuat 1: K-Nearest Neighbors (K-NN)
  + Nguyen ly: Do khoang cach Euclidean giua dac trung anh moi voi tap mau, chon nhan chiem da so trong k lang gieng gan nhat.
  + Uu diem: Don gian, khong can huan luyen phuc tap.
  + Nhuoc diem: Tinh toan cham tren video realtime, rat nhay cam voi su thay doi do sang ban ngay va quang sang ban dem.
- Giai thuat 2: Gaussian Naive Bayes (GNB)
  + Nguyen ly: Ap dung dinh ly xac suat Bayes voi gia dinh cac dac trung mau doc lap co dieu kien.
  + Uu diem: Toc do huan luyen va suy luan cuc nhanh (duoi 1 ms), ton it bo nho.
  + Nhuoc diem: Gia dinh cac kenh mau doc lap khong phu hop voi anh thuc te (mau sac phu thuoc chat che vao cuong do sang), dan den do chinh xac thap nhat khi bi nguoc sang.
- Giai thuat 3: Support Vector Machine (SVM voi RBF Kernel)
  + Nguyen ly: Tim sieu phang phan tach toi uu bien (Maximum Margin Hyperplane) trong khong gian dac trung phi tuyen.
  + Uu diem: Do chinh xac cao nhat trong cac giai thuat ML co dien tren tap du lieu nho, suy luan nhanh sau khi tim duoc Support Vectors.
  + Nhuoc diem: Phu thuoc hoan toan vao chat luong trich xuat dac trung thu cong; de bi lech khi goc quay camera xa lam meo mo dac trung HOG.
- Giai thuat 4: Random Forest (RF)
  + Nguyen ly: Tap hop nhieu cay quyet dinh (Ensemble of Decision Trees) ket hop ky thuat Bootstrap va Feature Subsampling.
  + Uu diem: Khang overfitting tot, cho phep danh gia do quan trong cua tung dac trung mau sac.
  + Nhuoc diem: Mo hinh lon hon cay don le, khong tu hoc duoc phan bo khong gian nhu mang tich chap.

3. Mo hinh doi chung Deep Learning:
- Mang no-ron tich chap (Mini-CNN / MobileNetV3 hoac YOLO classification):
  + Tu dong hoc dac trung khong gian va mau sac (End-to-End) khong can trich xuat dac trung thu cong.
  + Thich nghi tot voi hien tuong choi sang, quang sang va goc camera bi nghieng.

4. Bang chi so danh gia khoa hoc xuat ra tu thuc nghiem:
--------------------------------------------------------------------------------
Thuat toan    | Accuracy | Macro Precision | Macro Recall | Macro F1 | Latency (ms)
--------------------------------------------------------------------------------
K-NN          | Do thuc te | Do thuc te    | Do thuc te   | Do thuc te | Do thuc te
Naive Bayes   | Do thuc te | Do thuc te    | Do thuc te   | Do thuc te | Do thuc te
SVM (RBF)     | Do thuc te | Do thuc te    | Do thuc te   | Do thuc te | Do thuc te
Random Forest | Do thuc te | Do thuc te    | Do thuc te   | Do thuc te | Do thuc te
Deep Learning | Do thuc te | Do thuc te    | Do thuc te   | Do thuc te | Do thuc te
--------------------------------------------------------------------------------
Xuat ma tran nham lan (Confusion Matrix) cho tung giai thuat de phan tich cac truong hop bat sai.

5. So sanh ly thuyet cho bai toan phat hien phuong tien (Vehicle Detection):
- HOG ket hop Linear SVM (Dalal & Triggs 2005) va Sliding Window: Tuan tu quet qua tung vung anh, toc do chi dat 3-10 FPS tren CPU, rat de bao sai khi cac xe chen chuc che khuat nhau.
- Haar Cascades voi AdaBoost (Viola-Jones): Thich hop phat hien vat the co ti le co dinh, kem khi phat hien xe may va goc quay nghieng.
- YOLO (v8s / v11s): Xu ly toan bo anh trong 1 luot suy luan (One-stage), tu dong hoc dac trung da ty le, dat 30-60+ FPS tren GPU RTX 3050.

================================================================================
PHAN 5: KE HOACH DU LIEU VA THIET KE THUC NGHIEM KHOA HOC
================================================================================

5.1. Chien luoc thu thap va tong hop du lieu

1. Du lieu phuong tien:
- Nguon co san: Tap du lieu MS-COCO (cac lop car, motorcycle, bus, truck), tap BDD100K hoac cac bo dataset giao thong Viet Nam ma khoa da cung cap.
- Du lieu thu thap thuc te: Quay video tai 2-3 nut giao thong thuc te co den tin hieu (do phan giai 1080p, goc quay tu tren cao cheo xuong bao quat vach dung va cot den).
2. Du lieu den tin hieu:
- Tap du lieu S2TLD hoac LISA Traffic Light Dataset ket hop mau cat thuc te tu video nut giao thong.

5.2. Nguyen tac chia tap du lieu tranh ro ri (Data Leakage Prevention)

- NGUYEN TAC BAT BUOC: Khong bao gio chia tap Train/Val/Test bang cach xao tron ngau nhien cac khung hinh (Random Frame Shuffle) duoc cat tu cung mot video clip. Dieu nay gay ra hien tuong ro ri thong tin tram trong va danh gia ao ve do chinh xac.
- QUY TAC CHIA: Chia theo chuoi video (Video-level split).
  + Tap Train: 70% thoi luong video (cac nut giao A, B o cac thoi diem khac nhau).
  + Tap Validation: 15% thoi luong video (dung de tune hyperparameter va early stopping).
  + Tap Test doc lap: 15% thoi luong video (nut giao moi hoac doan video hoan toan doc lap chua tung huan luyen).

5.3. He thong chi so danh gia khoa hoc

1. Danh gia Detector phuong tien va den:
- Precision, Recall, F1-score tai IoU threshold 0.5.
- mAP@0.5 va mAP@0.5:0.95.
2. Danh gia Tracker:
- IDF1 (Identification F1-score): Do do nhat quan cua ID phuong tien.
- HOTA (Higher Order Tracking Accuracy): Chi so toan dien hien dai nhat.
- ID Switches: So lan bi doi ID tren mot quy dao di chuyen.
3. Danh gia He thong Phat hien Vi pham (Event-Level Evaluation):
- True Positive (TP): Xe thuc su vuot den do va he thong canh bao dung.
- False Positive (FP): Xe khong vuot (dung sat vach, re phai cho phep) nhung he thong bao vi pham (Loi nghiem trong can giam toi thieu).
- False Negative (FN): Xe vuot den do nhung he thong bo sot.
- Event Precision = TP / (TP + FP)
- Event Recall = TP / (TP + FN)
- Event F1-score = (2 x Precision x Recall) / (Precision + Recall)
4. Danh gia Hieu nang Thoi gian thuc tren Laptop:
- Thoi gian xu ly trung binh moi khung hinh (Latency ms/frame).
- Toc do khung hinh xu ly (FPS).
- Luong bo nho VRAM va RAM chiem dung.

================================================================================
PHAN 6: KIEN TRUC HE THONG VA LUONG XU LY (AI PIPELINE)
================================================================================

6.1. So do khoi kien truc tong the

Nguon Video Dau vao (File MP4 / Camera RTSP)
         |
         v
[Module 1: Preprocessing & Frame Management]
         |
         +-------------------------------------+
         |                                     |
         v                                     v
[Module 2: Traffic Light Detection]   [Module 3: Vehicle Detection]
   (YOLOv11 / HSV Analysis)              (YOLOv11s Detector)
         |                                     |
         v                                     v
[Trang thai: Red / Yellow / Green]    [Module 4: Multi-Object Tracker]
         |                               (ByteTrack / BoT-SORT)
         |                                     |
         |                            [Track ID & Bounding Box]
         |                                     |
         +------------------+------------------+
                            |
                            v
           [Module 5: Spatio-Temporal Violation Logic]
              - Xac dinh toa do tiep xuc banh xe
              - Kiem tra vector cat vach dung (Stop Line)
              - Xac nhan trang thai den do duy tri
              - Bo dem thoi gian va vach dem thu 2
                            |
                            +------------------+
                            |                  |
                       (Co vi pham)       (Hop le)
                            |                  |
                            v                  v
               [Module 6: Evidence Extractor]  (Bo qua)
                  - Luu 3 frame bang chung
                  - Cat video ngan 5s
                  - Nhan dang bien so (OCR neu co)
                            |
                            v
               [Module 7: Single Demo UI]
                  - Hien thi truc tiep tren man hinh
                  - Bieu dien vach ROI, box, trace quy dao
                  - Bang danh sach su kien vi pham co the bam vao xem lai

6.2. Dinh nghia ro rang cac giao dien modul (Interfaces)

- Module 1 (Input): Nhan duong dan video, doc khung hinh bang OpenCV VideoCapture, resize chuan hoa kich thuoc theo cau hinh.
- Module 2 (Light State): Dau ra la dict { "state": "RED" | "YELLOW" | "GREEN" | "UNKNOWN", "confidence": float, "box": [x1, y1, x2, y2] }.
- Module 3 va 4 (Detection & Tracking): Dau ra la list cac doi tuong [ { "track_id": int, "class_name": str, "box": [x1, y1, x2, y2], "confidence": float, "bottom_center": (x, y) } ].
- Module 5 (Violation Logic): Dau ra la su kien vi pham { "violation_id": str, "track_id": int, "vehicle_type": str, "timestamp": float, "frame_index": int, "status": "CONFIRMED" }.
- Module 6 (Evidence): Dau ra la thu muc bang chung gom evidence_info.json, frame_before.jpg, frame_crossing.jpg, frame_after.jpg, clip.mp4.
- Module 7 (Giao dien): Nhan khung hinh da duoc ve overlay tu pipeline va hien thi len cua so giao dien GUI duy nhat.

================================================================================
PHAN 7: CHI TIET KE HOACH 9 GIAI DOAN THUC HIEN
================================================================================

--------------------------------------------------------------------------------
GIAI DOAN 0: KIEM KE VA CHUAN HOA MOI TRUONG
--------------------------------------------------------------------------------
1. Muc tieu:
- Kiem ke toan bo phan cung, he dieu hanh, dung luong o dia, tai lieu hien co.
- Thiet lap moi truong lap trinh Python ao (virtual environment), cai dat PyTorch phien ban phu hop CUDA 12.x/13.0, OpenCV, Ultralytics tren laptop ca nhan.
2. Cong viec thuc hien:
- Khoi tao moi truong ao Python: python -m venv venv_nln
- Cai dat PyTorch ho tro GPU phu hop card RTX 3050.
- Cai dat cac thu vien phu tro: ultralytics, opencv-python, numpy, pandas, matplotlib, filterpy, lapx.
- Viet script nho test_env.py kiem tra nhan dien GPU, do thoi gian suy luan 1 batch thu nghiem.
- Tao cau truc thu muc du an chuan hoa.
3. Kien thuc can nghien cuu:
- Quan ly moi truong Python tren Windows.
- Tuong thich giua CUDA Toolkit, cuDNN va PyTorch.
4. Cong nghe, thu vien:
- Python 3.11, PyTorch (CUDA build), Ultralytics, PowerShell.
5. Dau vao va San pham ban giao:
- Dau vao: Cau hinh laptop thuc te.
- San pham ban giao: File requirements.txt, script test_env.py chay thanh cong bao GPU san sang.
6. Tieu chi nghiem thu:
- Lenh torch.cuda.is_available() tra ve True va hien thi dung NVIDIA GeForce RTX 3050 Laptop GPU.
7. Rui ro va phuong an xu ly:
- Rui ro: Xung dot phien ban CUDA tren Windows.
- Xu ly: Cai dat PyTorch theo dung cu phap ho tro CUDA tu trang chu pytorch.org.
8. Checklist trang thai:
- [X] Kiem ke phan cung may tinh (Da hoan thanh - RTX 3050 6GB, Driver 581.86, CUDA 13.0)
- [X] Tao cay thu muc du an chuan hoa (Da hoan thanh - 17 thu muc tren D:\NIENLUANNGANH)
- [X] Tao file requirements.txt toi uu (Da hoan thanh - khong co thu vien web enterprise)
- [X] Tao file config.yaml cau hinh trung tam (Da hoan thanh)
- [X] Tao skeleton ma nguon src/ (Da hoan thanh - 5 module: detector, tracker, traffic_light, violation_logic, evidence)
- [X] Tao moi truong ao Python 3.11 tai D:\NIENLUANNGANH\venv_nln (Da hoan thanh)
- [X] Cai dat PyTorch CUDA 12.4 vao venv_nln (Da hoan thanh - PyTorch 2.6.0+cu124)
- [X] Cai dat day du cac thu vien AI (Da hoan thanh - Ultralytics, OpenCV, Scikit-learn, LapX, FilterPy, PyQt6...)
- [X] Chay test_env.py kiem tra toan dien (Da hoan thanh - Nhan dung RTX 3050 6GB Laptop GPU, CUDA=True, 9.3ms matmul, ~454 FPS inference)
Cap nhat: 2026-11-05 - Giai doan 0 DA HOAN THANH 100%. He thong da san sang chuyen sang Giai doan 1.
9. Uoc luong thoi gian: 1 - 2 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 1: NGHIEN CUU TONG QUAN VA THIET LAP CAU HOI KHOA HOC
--------------------------------------------------------------------------------
1. Muc tieu:
- Tong hop co so ly thuyet ve Object Detection, Multi-Object Tracking, va xu ly hinh hoc khong gian - thoi gian trong giam sat giao thong.
- Xac lap ro rang cac cau hoi nghien cuu va khoang trong hoc thuat can giai quyet.
2. Cong viec thuc hien:
- Tim doc cac bai bao goc ve YOLOv8, YOLOv11, ByteTrack (ECCV 2022), BoT-SORT (2022).
- Tong hop cac nghien cuu ve giam sat giao thong tai Viet Nam va DH Can Tho.
- Viet ban tong quan ly thuyet khoa hoc, lap bang so sanh dac trung giua cac phuong phap.
3. Kien thuc can nghien cuu:
- Co che Non-Maximum Suppression (NMS), C3k2 block, C2PSA attention.
- Thuat toan Kalman Filter va Hungarian Algorithm trong tracking doi tuong.
4. Cong nghe va tai lieu:
- Cac bai bao tren arXiv, IEEE Xplore, Tap chi Khoa hoc DH Can Tho.
5. Dau vao va San pham ban giao:
- San pham ban giao: File bao_cao_tong_quan_khoa_hoc.md voi he thong trich dan hoc thuat day du.
6. Tieu chi nghiem thu:
- Co it nhat 10 tai lieu tham khao uy tin, co phan tich ro uu nhuoc diem cua tung phuong phap.
7. Rui ro va phuong an xu ly:
- Rui ro: Sa da vao ly thuyet qua rong khong sat de tai.
- Xu ly: Chi tap trung vao Detection, Tracking va phat hien trang thai den giao thong.
8. Checklist trang thai:
- [X] Tong hop tai lieu ve YOLOv8 va YOLOv11 (Da hoan thanh - So sanh kien truc C2f vs C3k2, C2PSA, params, mAP)
- [X] Tong hop tai lieu ve ByteTrack va BoT-SORT (Da hoan thanh - Phan tich chi tiet D_low, CMC Lucas-Kanade+RANSAC, IDF1 77.3% vs 78.7%)
- [X] Xay dung bo cau hoi nghien cuu ro rang (Da hoan thanh - 5 RQ: Detection, Tracking, Traffic Light, Violation Logic, System FPS)
Cap nhat: 2026-11-10 - Giai doan 1 DA HOAN THANH 100%. File bao_cao_tong_quan_khoa_hoc.md da tao voi 13 tai lieu tham khao, 5 research gaps dac thu giao thong Viet Nam.
9. Uoc luong thoi gian: 3 - 5 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 2: DULIEU VA THIET KE THUC NGHIEM KHOA HOC
--------------------------------------------------------------------------------
1. Muc tieu:
- Thu thap, loc, to chuc va chuan hoa tap du lieu hinh anh va video giao thong.
- Xay dung tap kiem thu Test Set doc lap dung tieu chuan tranh ro ri du lieu.
2. Cong viec thuc hien:
- Thu thap hoac tong hop 3-5 video chat luong cao tu nut giao thong thuc te.
- Trinh bay ro rang danh sach lop can phat hien: car, motorcycle, truck, bus, traffic_light.
- Chia tap du lieu theo chuoi video: Train (70%), Val (15%), Test (15%).
- Viet script kiem tra tinh toan ven cua nhan (Label Verification) va thong ke phan bo lop.
3. Kien thuc can nghien cuu:
- Dinh dang nhan YOLO (txt x_center y_center width height).
- Ky thuat tang cuong du lieu (Data Augmentation): Mosaic, Mixup, HSV shift, Random Scale.
4. Cong nghe, cong cu:
- Roboflow hoac LabelImg hoac CVAT, script Python xu ly du lieu.
5. Dau vao va San pham ban giao:
- San pham: Thu muc dataset/ tren o dia D voi file data.yaml, thong ke phan bo so luong mau.
6. Tieu chi nghiem thu:
- Khong co loi nhan thieu, khong co du lieu bi trung lap giua Train va Test.
7. Rui ro va phuong an xu ly:
- Rui ro: Thieu nhan xe may dac thu Viet Nam tren tap COCO mac dinh.
- Xu ly: Thu thap bo sung 300 - 500 anh xe may thuc te Viet Nam de fine-tune bo sung.
8. Checklist trang thai:
- [ ] Thu thap video giao thong thuc te
- [ ] Chuan hoa format nhan va chia tap Train/Val/Test
- [ ] Viet script kiem tra va thong ke du lieu
9. Uoc luong thoi gian: 4 - 6 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 3: THIET KE MO-DUN VA KIEN TRUC PIPELINE
--------------------------------------------------------------------------------
1. Muc tieu:
- Thiet ke kien truc pipeline huong doi tuong (OOP), tach biet ro rang tung module.
- Dinh nghia ro rang cac lop (Classes), ham (Methods) va cau truc du lieu trao doi.
2. Cong viec thuc hien:
- Xay dung lop VideoReader, Detector, Tracker, TrafficLightClassifier, ViolationDetector, Visualizer.
- Thiet lap file cau hinh trung tam config.yaml quan ly toan bo nguong (thresholds, fps, toa do ROI).
- Ve so do tuan tu (Sequence Diagram) va so do kien truc luong du lieu.
3. Kien thuc can nghien cuu:
- Design Pattern (Pipeline Pattern, Strategy Pattern cho Tracker).
4. Cong nghe:
- Python dataclasses, PyYAML, OpenCV.
5. Dau vao va San pham ban giao:
- San pham: Bo khung ma nguon src/ gom cac file module rong co day du docstring, tham so va kieu du lieu.
6. Tieu chi nghiem thu:
- Cac module doc lap, de dang thay the Tracker (ByteTrack sang BoT-SORT) ma khong sua code logic vi pham.
7. Rui ro va phuong an xu ly:
- Rui ro: Code bi gan ket qua chat (tightly coupled).
- Xu ly: Su dung cau truc du lieu tieu chuan dict hoac dataclass truyen qua tung frame.
8. Checklist trang thai:
- [ ] Thiet ke file cau hinh config.yaml
- [ ] Dinh nghia cac lop interface trong thu muc src/
- [ ] Hoan thien tai lieu mo ta luong du lieu
9. Uoc luong thoi gian: 2 - 3 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 4: XAY DUNG BASELINE PIPELINE TOI THIEU
--------------------------------------------------------------------------------
1. Muc tieu:
- Xay dung mot pipeline toi thieu co the chay tu dau vao video den dau ra hien thi nhan dien co ban.
- Dung lam moc doi chung (Baseline) de danh gia cac cai tien sau nay.
2. Cong viec thuc hien:
- Su dung model pretrained YOLOv8s de phat hien phuong tien mac dinh.
- Tich hop ByteTrack co san de danh so ID cho xe qua cac frame.
- Viet ham ve Bounding Box va ID len video bang OpenCV.
- Do toc do xu ly FPS ban dau tren GPU RTX 3050.
3. Kien thuc can nghien cuu:
- OpenCV VideoWriter, cv2.putText, cv2.rectangle.
4. Cong nghe:
- PyTorch, Ultralytics, OpenCV.
5. Dau vao va San pham ban giao:
- San pham: Script run_baseline.py xuat ra file video annotated_output.mp4.
6. Tieu chi nghiem thu:
- Script chay khong loi tren doan video test 30 giay, hien thi xe co khung bao va ID on dinh.
7. Rui ro va phuong an xu ly:
- Rui ro: FPS bi giam do ham ve hinh anh cham.
- Xu ly: Toi uu hoa cac thao tac ve tren frame numpy.
8. Checklist trang thai:
- [ ] Cai dat script chay baseline voi YOLOv8s
- [ ] Tich hop ByteTrack co ban
- [ ] Ghi nhan FPS va tieu ton VRAM moc dau tien
9. Uoc luong thoi gian: 2 - 3 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 5: THUC NGHIEM, HUAN LUYEN VA TOI UU HOA MO HINH
--------------------------------------------------------------------------------
1. Muc tieu:
- Huan luyen / fine-tune cac mo hinh ung vien (YOLOv8s, YOLOv11s) tren tap du lieu giao thong.
- Xay dung thuc nghiem doi chung so sanh 4 giai thuat Hoc may truyen thong (K-NN, Naive Bayes, SVM, Random Forest) voi Hoc sau (Deep Learning) tren bai toan phan loai den tin hieu.
- So sanh khach quan ket qua giua cac mo hinh va chon ra mo hinh toi uu nhat cho he thong.
2. Cong viec thuc hien:
- Chay huan luyen YOLO voi cac sieu tham so co dinh (epochs=50, imgsz=640, batch=16, amp=True).
- Ghi nhan log huan luyen (Loss, mAP@0.5, mAP@0.5:0.95 qua tung epoch).
- Viet script benchmark_ml_vs_dl.py su dung scikit-learn:
  + Trich xuat dac trung Color Histogram (HSV) va HOG tu tap anh crop den tin hieu.
  + Huan luyen va danh gia 4 mo hinh: KNeighborsClassifier, GaussianNB, SVC (RBF), RandomForestClassifier.
  + Tinh toan Accuracy, Macro Precision, Macro Recall, Macro F1-score, do thoi gian suy luan (ms/anh).
  + Ve va xuat ma tran nham lan (Confusion Matrix) cua tung thuat toan.
- So sanh thuc nghiem giua ByteTrack va BoT-SORT ve ID Switches va do on dinh quy dao.
3. Kien thuc can nghien cuu:
- Scikit-learn classification metrics, confusion matrix visualization.
- Transfer learning, Loss function (CIoU, DFL), ky thuat giam learning rate (Cosine Annealing).
4. Cong nghe:
- Scikit-learn, Ultralytics training API, Matplotlib / Seaborn ve bieu do so sanh.
5. Dau vao va San pham ban giao:
- San pham: Checkpoint model tot nhat best.pt, script benchmark_ml_vs_dl.py, file ket_qua_thuc_nghiem.md chua bang so sanh va bieu do ma tran nham lan.
6. Tieu chi nghiem thu:
- Script benchmark chay tu dong xuat day du so lieu 4 giai thuat ML va Deep Learning.
- Mo hinh chon co mAP@0.5 tren tap test cao hon baseline va dat toc do tren 30 FPS tren RTX 3050.
7. Rui ro va phuong an xu ly:
- Rui ro: Tap anh crop den tin hieu bi mat can bang so luong giua cac lop mau.
- Xu ly: Can bang du lieu bang ky thuat Random Over-sampling hoac gan trong so lop (class weights).
8. Checklist trang thai:
- [ ] Fine-tune YOLOv11s tren tap du lieu
- [ ] Chay benchmark 4 giai thuat ML (KNN, Naive Bayes, SVM, RF) va xuat Confusion Matrix
- [ ] So sanh mAP giua YOLOv8s va YOLOv11s
- [ ] So sanh thuc nghiem ByteTrack va BoT-SORT
9. Uoc luong thoi gian: 4 - 6 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 6: XAC DINH SU KIEN VI PHAM VA XU LY CAC CA KHO
--------------------------------------------------------------------------------
1. Muc tieu:
- Xay dung thuat toan phan doan vi pham dua tren hinh hoc khong gian va chuoi thoi gian.
- Xu ly triet de cac truong hop bien de giam False Positive xuong muc thap nhat.
2. Cong viec thuc hien:
- Cai dat thuat toan xac dinh vi tri banh xe tiep dat dua vao tam day bounding box.
- Cai dat thuat toan kiem tra giao cat doan thang (Line Intersection Algorithm) giua vector di chuyen cua xe va Vach dung (Stop Line).
- Thiet lap co che 2 vach ao: Vach 1 (canh bao), Vach 2 (xac nhan vi pham sau khi di vao giao lo).
- Tich hop bo dem thoi gian den do: Chi kich hoat vi pham khi den do da sang on dinh it nhat 1.0 giay (tranh bat loi xe dung khong kip khi den vang chuyen do).
- Viet module EvidenceExtractor: Tu dong trich xuat 3 anh bang chung kem clip 5 giay khi xac nhan vi pham.
3. Kien thuc can nghien cuu:
- Hinh hoc tinh toan (Computational Geometry), vector cross product, spatio-temporal buffer.
4. Cong nghe:
- Python, Shapely / NumPy, OpenCV.
5. Dau vao va San pham ban giao:
- San pham: Module violation_engine.py va evidence_saver.py hoan chinh.
6. Tieu chi nghiem thu:
- Tren video test, bat dung cac xe vuot qua giao lo khi den do, khong bat nham xe dung ngay sat vach.
7. Rui ro va phuong an xu ly:
- Rui ro: Nhay ID do che khuat lam sai lech quy dao.
- Xu ly: Kiem tra do dai quy dao (Track history length > 10 frames) moi cho phep danh gia vi pham.
8. Checklist trang thai:
- [ ] Cai dat thuat toan giao cat vach dung theo thoi gian
- [ ] Thiet lap co che 2 vach kiem chung va bo loc den vang/do
- [ ] Cai dat module trich xuat bang chung tu dong
9. Uoc luong thoi gian: 3 - 5 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 7: HOAN THIEN DUY NHAT MOT GIAO DIEN DEMO
--------------------------------------------------------------------------------
1. Muc tieu:
- Xay dung duy nhat mot giao dien demo tien dung, hien dai, chuyen nghiep, phu hop bao cao nien luan.
- Tuyet doi khong dung kien truc web nhieu tang phuc tap; chon Desktop GUI nhe hoac Local Webview.
2. Cong viec thuc hien:
- Thiet ke giao dien truc quan (khuyen nghi dung PyQt6 / PySide6 hoac Streamlit cuc bo):
  + Khung hinh video playback hien thi realtime bounding box, track ID, trace quy dao va trang thai den.
  + Bang hien thi danh sach su kien vi pham (Thoi gian, Loai xe, ID, Tinh trang).
  + Khung xem nhanh bang chung khi click vao mot su kien vi pham (Hien thi 3 frame va anh crop).
  + Bang dieu khien don gian: Chon file video, Dat vach dung (bang chuot), Nut Play/Pause/Stop.
3. Kien thuc can nghien cuu:
- Lap trinh GUI voi PyQt6 (QThread tranh treo UI khi inference) hoac Streamlit component.
4. Cong nghe:
- PyQt6 / PySide6 (hoac Streamlit cuc bo tren trinh duyet).
5. Dau vao va San pham ban giao:
- San pham: File main_app.py chay truc tiep bang lenh python main_app.py.
6. Tieu chi nghiem thu:
- Giao dien chay muot ma, khong gay drop FPS cua pipeline AI, thao tac click xem bang chung tuc thi.
7. Rui ro va phuong an xu ly:
- Rui ro: Pipeline AI chay tren cung luong voi GUI lam dong bang giao dien.
- Xu ly: Su dung QThread hoac Worker Thread tach biet giua luong doc/suy luan video va luong ve giao dien.
8. Checklist trang thai:
- [ ] Thiet ke layout giao dien duy nhat
- [ ] Tich hop Worker Thread ket noi pipeline AI voi GUI
- [ ] Tich hop chuc nang click xem lai bang chung vi pham
9. Uoc luong thoi gian: 3 - 5 ngay.

--------------------------------------------------------------------------------
GIAI DOAN 8: DANH GIA TONG THE, THUC NGHIEM TOAN DIEN VA VIET BAO CAO
--------------------------------------------------------------------------------
1. Muc tieu:
- Do luong toan bo chi so tren tap test doc lap va tong hop ket qua nghien cuu khoa hoc.
- Hoan thien bao cao nien luan chi tiet, trung thuc, khoa hoc va slide bao cao.
2. Cong viec thuc hien:
- Chay danh gia toan bo he thong tren tap video test: Tinh Precision, Recall, F1 cua su kien vi pham.
- Thuc hien Ablation Study: Danh gia vai tro cua bo loc thoi gian, vai tro cua vach ao so 2, so sanh ByteTrack vs BoT-SORT.
- Tong hop bang bieu, do thi, hinh anh bang chung truc quan.
- Hoan thien cac chuong cua quyen bao cao: Chuong 1 (Tong quan), Chuong 2 (Co so ly thuyet), Chuong 3 (Thiet ke he thong), Chuong 4 (Thuc nghiem va danh gia), Chuong 5 (Ket luan va Huong phat trien).
3. Kien thuc can nghien cuu:
- Trinh bay bao cao khoa hoc theo chuan hoc thuat.
4. Cong nghe:
- LaTeX hoac Microsoft Word, Matplotlib, Seaborn ve bieu do khoa hoc.
5. Dau vao va San pham ban giao:
- San pham: File bao cao hoan chinh (.pdf/.docx), file slide thuyet trinh, ma nguon hoan thien tren kho luu tru.
6. Tieu chi nghiem thu:
- Bao cao trinh bay day du so lieu thuc nghiem, co phan tich ro cac truong hop bat sai (Error Analysis) va gioi han cua he thong.
7. Rui ro va phuong an xu ly:
- Rui ro: So lieu danh gia khong dat nhu ky vong ban dau.
- Xu ly: Phan tich nguyen nhan khoa hoc mot cach khach quan (do goc camera, do che khuat), neu ro huong khac phuc trong tuong lai.
8. Checklist trang thai:
- [ ] Chay thuc nghiem toan he thong va do chi so Event F1
- [ ] Thuc hien Ablation Study va ve bieu do
- [ ] Hoan thien quyen bao cao va slide trinh chieu
9. Uoc luong thoi gian: 5 - 7 ngay.

================================================================================
PHAN 8: BO MAU PROMPT TRIEN KHAI CHO TUNG GIAI DOAN
================================================================================

Duoi day la cac prompt duoc thiet ke san. Sinh vien chi can sao chep nguyen van prompt cua tung giai doan va gui cho AI de bat dau thuc hien.

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 0: KIEM KE VA THIET LAP MOI TRUONG
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 0]
Vai tro: Chuyen gia he thong va ky su AI.
Boi canh: Toi dang thuc hien de tai Nien luan ve He thong tu dong phat hien phuong tien vi pham den tin hieu giao thong tren may laptop Windows 11 co card NVIDIA GeForce RTX 3050 6GB Laptop GPU, CPU Intel i5-12450HX, 12GB RAM, o D con trong 170GB.
Nhiem vu cua ban:
1. Huong dan tao moi truong ao Python 3.11 chuan tren o dia D.
2. Cung cap lenh cai dat PyTorch phien ban phu hop CUDA cua he thong.
3. Tao file requirements.txt toi uu cho laptop gom cac thu vien AI can thiet (ultralytics, opencv, numpy, lapx...), tuyet doi khong them cac thu vien web enterprise nang nhe.
4. Viet mot script test_env.py de kiem tra toan dien: nhan dien GPU, dung luong VRAM, thoi gian chay thu nghiem mot tensor tren GPU.
5. Tao cay thu muc chuan cho toan bo ma nguon du an.
Tieu chi nghiem thu: Script test_env.py chay thanh cong, nhan dung RTX 3050.
Cap nhat lai checklist cua Giai doan 0 sau khi thuc hien xong.
[KET THUC PROMPT GIAI DOAN 0]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 1: NGHIEN CUU TONG QUAN VA CO SO LY THUYET
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 1]
Vai tro: Chuyen gia nghien cuu khoa hoc Thi giac may tinh va AI.
Boi canh: De tai "Xay dung he thong tu dong phat hien phuong tien vi pham hieu lenh den tin hieu giao thong".
Nhiem vu cua ban:
1. Soan thao noi dung tong quan co so ly thuyet chat che, tap trung vao:
   - Co che hoat dong cua mo hinh YOLO (so sanh kien truc YOLOv8 va YOLOv11 ve backbone, neck, attention C2PSA).
   - Thuat toan theo doi doi tuong da muc tieu: So sanh chi tiet ByteTrack (lien ket box diem thap) va BoT-SORT (bu chuyen dong camera CMC).
   - Cac phuong phap xac dinh trang thai den tin hieu giao thong (Object Detection truc tiep vs Crop phan tich mau HSV).
2. Trinh bay it nhat 8 cong trinh khoa hoc co trinh dan day du (Ten bai bao, Tac gia, Nam, Noi xuat ban/arXiv, Dong gop chinh).
3. Phan tich ro rang khoang trong nghien cuu khi ap dung vao giao thong hon hop nhieu xe may tai Viet Nam.
Tieu chi nghiem thu: Noi dung mang tinh hoc thuat cao, trinh dan xac thuc, phan biet ro kien thuc da cong bo va de xuat cua du an.
Cap nhat checklist Giai doan 1 sau khi hoan thanh.
[KET THUC PROMPT GIAI DOAN 1]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 2: DULIEU VA THIET KE THUC NGHIEM KHOA HOC
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 2]
Vai tro: Chuyen gia xu ly du lieu va thiet ke thuc nghiem AI.
Boi canh: Du an phat hien phuong tien vi pham den giao thong can tap du lieu chuan hoa de huan luyen va danh gia.
Nhiem vu cua ban:
1. Xay dung quy trinh to chuc du lieu tren o dia D tranh ro ri du lieu (Data Leakage) theo phuong phap Video-level split.
2. Thiet ke file cau hinh dataset data.yaml theo format YOLO chuan (cac lop: car, motorcycle, bus, truck, traffic_light).
3. Viet script Python data_checker.py de kiem tra toan ven file anh, file nhan, kich thuoc bounding box hop le va thong ke tan suat xuat hien cac lop.
4. De xuat chien luoc tang cuong du lieu (Data Augmentation) phu hop voi camera giao thong ngoai troi (thay doi do sang, anh huong thoi tiet).
5. Xac lap tieu chuan danh gia: mAP@0.5, mAP@0.5:0.95, HOTA, IDF1 va Event-level F1-score.
Tieu chi nghiem thu: Script chay kiem tra du lieu khong loi, tap du lieu duoc to chuc ro rang.
Cap nhat checklist Giai doan 2 sau khi hoan thanh.
[KET THUC PROMPT GIAI DOAN 2]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 3: THIET KE KIEN TRUC PIPELINE
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 3]
Vai tro: Kien truc su phan mem AI he thong.
Boi canh: Can thiet ke kien truc module ro rang, de bao tri va mo rong cho pipeline AI phat hien vi pham den do.
Nhiem vu cua ban:
1. Thiet ke file cau hinh tap trung config.yaml (chua tat ca tham so: duong dan model, nguong confidence, toa do vach dung, thoi gian buffer, duong dan luu bang chung).
2. Xay dung cau truc thu muc src/ theo mau Module huong doi tuong (OOP):
   - video_loader.py: Doc va chuan hoa luong video.
   - detector.py: Wrapper cho mo hinh YOLO voi co che bat tat FP16.
   - tracker.py: Interface chung cho phep de dang doi giua ByteTrack va BoT-SORT.
   - traffic_light.py: Nhan dien trang thai den.
   - violation_logic.py: Xu ly hinh hoc va logic thoi gian cat vach dung.
   - evidence.py: Trich xuat va quan ly file bang chung.
3. Viet code khung (skeleton code) cho tat ca cac file tren voi day du Type Hints va docstring giai thich ro Input/Output.
Tieu chi nghiem thu: Cac module khong bi loi import cheo, giao dien ket noi ro rang.
Cap nhat checklist Giai doan 3 sau khi hoan thanh.
[KET THUC PROMPT GIAI DOAN 3]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 4: TRIEN KHAI BASELINE PIPELINE
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 4]
Vai tro: Ky su thi giac may tinh (Computer Vision Engineer).
Boi canh: Can co mot ban chay toi thieu (Baseline) de kiem chung toan bo luong xu ly tu video dau vao den video dau ra.
Nhiem vu cua ban:
1. Viet ma nguon hoan chinh cho run_baseline.py tich hop model YOLOv8s pretrained va ByteTrack.
2. Trien khai luong doc video, suy luan detection, cap nhat tracker va ve bounding box kem ID len frame.
3. Tinh toan va in ra man hinh toc do FPS thuc te, do tre trung binh ms/frame tren GPU RTX 3050.
4. Xuat video ket qua co ve overlay ra file annotated_baseline.mp4.
Tieu chi nghiem thu: Script chay muot ma tren video test 30s khong vang loi, FPS dat tren 25 FPS tren GPU.
Cap nhat checklist Giai doan 4 sau khi hoan thanh.
[KET THUC PROMPT GIAI DOAN 4]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 5: THUC NGHIEM, HUAN LUYEN VA TOI UU HOA
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 5]
Vai tro: Chuyen gia Deep Learning, Machine Learning va Toi uu thuat toan.
Boi canh: Cung cap bang chung thuc nghiem khoa hoc so sanh giua cac mo hinh deep learning va giai thuat hoc may truyen thong tren tap test doc lap.
Nhiem vu cua ban:
1. Viet script train.py tinh chinh (fine-tune) YOLOv11s va YOLOv8s tren dataset giao thong voi cau hinh an toan VRAM 6GB (FP16, batch=8 hoac 16).
2. Viet script evaluate.py do luong mAP@0.5 va mAP@0.5:0.95 tren tap Test cho ca hai model YOLO.
3. Viet script benchmark_ml_vs_dl.py su dung scikit-learn:
   - Trich xuat dac trung Color Histogram (HSV) va HOG tu tap anh crop den tin hieu giao thong.
   - Huan luyen va danh gia 4 mo hinh co dien: KNeighborsClassifier, GaussianNB, SVC (RBF kernel), RandomForestClassifier.
   - So sanh voi mo hinh Deep Learning tren cac chi so: Accuracy, Macro Precision, Macro Recall, Macro F1-score va do tre suy luan (ms/anh).
   - Xuat ma tran nham lan (Confusion Matrix) va luu ket qua vao bang ket_qua_so_sanh_ml_vs_dl.csv.
4. So sanh hieu nang tracking giua ByteTrack va BoT-SORT tren cung mot video thu nghiem (so sanh ID Switches va FPS).
5. Tao file ket_qua_thuc_nghiem.md chua toan bo bang so lieu thuc te thu duoc, nhan xet khach quan va ve bieu do minh hoa de dua thang vao bao cao nien luan.
Tieu chi nghiem thu: Co so lieu thuc nghiem that, danh gia khach quan giua ML co dien va Deep Learning.
Cap nhat checklist Giai doan 5 sau khi hoan thanh.
[KET THUC PROMPT GIAI DOAN 5]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 6: XAC DINH SU KIEN VI PHAM VA CA KHO
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 6]
Vai tro: Chuyen gia thuat toan hinh hoc va logic AI.
Boi canh: Can mot co che phat hien vuot den do chinh xac, loai bo false positive khi xe dung sat vach hoac re phai.
Nhiem vu cua ban:
1. Viet module violation_logic.py hoan chinh:
   - Dinh nghia vach dung va vach xac nhan (2 virtual lines).
   - Tinh toan diem tiep dat cua xe (bottom center) va vector quy dao qua cac frame.
   - Thuat toan xac dinh xe cat qua vach bang tich co huong (Vector Cross Product).
   - Bo dem thoi gian den do: chi bat loi khi den do da sang on dinh > 1 giay.
   - Loc huong xe re phai hop le dua tren goc huong di chuyen.
2. Viet module evidence.py: Tu dong luu thong tin su kien (JSON), 3 frame anh (truoc - trong - sau vuot vach) va doan video clip ngan 5 giay.
3. Viet script test_violation.py chay thu tren doan video co ca xe vi pham va xe dung dung vach.
Tieu chi nghiem thu: Phan biet chinh xac giua xe vuot den va xe chi dung cham vach roi dung han.
Cap nhat checklist Giai doan 6 sau khi hoan thanh.
[KET THUC PROMPT GIAI DOAN 6]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 7: HOAN THIEN GIAO DIEN DEMO DUY NHAT
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 7]
Vai tro: Ky su lap trinh giao dien he thong AI.
Boi canh: Xay dung duy nhat mot giao dien demo hien dai, tien dung de trinh dien toan bo pipeline phat hien vi pham, phuc vu bao cao nien luan.
Nhiem vu cua ban:
1. Xay dung mot giao dien Desktop duy nhat (su dung PyQt6 / PySide6 hoac Streamlit cuc bo):
   - Khung phat video voi overlay truc quan (box, ID, mau den tin hieu o goc man hinh, vach dung to mau do khi den do).
   - Bang danh sach su kien vi pham realtime (Thoi gian, ID phuong tien, Loai phuong tien, Trang thai xac nhan).
   - Bang xem chi tiet bang chung: Khi bam vao 1 dong vi pham, lap tuc hien 3 anh bang chung va nut phat video ngan vi pham do.
   - Cac nut dieu khien: Chon file video, Bat dau / Tam dung, Cau hinh vach dung truc quan bang click chuot.
2. Tuan thu nguyen tac: Giao dien nhe, dep, tach luong voi pipeline AI de khong lam giam FPS suy luan.
Tieu chi nghiem thu: Ung dung chay on dinh bang lenh python main.py, giao dien thao tac thuan tien, khong bi crash.
Cap nhat checklist Giai doan 7 sau khi hoan thanh.
[KET THUC PROMPT GIAI DOAN 7]

--------------------------------------------------------------------------------
PROMPT CHO GIAI DOAN 8: DANH GIA TONG THE VA HOAN THIEN BAO CAO
--------------------------------------------------------------------------------
Noi dung prompt:
[BAT DAU PROMPT GIAI DOAN 8]
Vai tro: Chuyen gia nghien cuu AI va Huong dan khoa hoc.
Boi canh: Giai doan cuoi cua Nien luan CNTT, can tong hop toan bo ket qua nghien cuu de viet bao cao hoan chinh.
Nhiem vu cua ban:
1. Huong dan va viet script danh gia toan dien tren tap test: Tinh Event-level Precision, Recall, F1-score cua he thong.
2. Thuc hien phan tich cac truong hop bat sai (Error Analysis): Giai thich nguyen nhan cac ca False Positive va False Negative duoi goc do thi giac may tinh.
3. Lap de cuong chi tiet va soan thao noi dung cac chuong cua quyen Bao cao Nien luan (Chuong 1 den Chuong 5).
   - Trong do Chuong 2 va Chuong 5 trinh bay chi tiet bang so sanh va bieu do thuc nghiem giua K-NN, Naive Bayes, SVM, Random Forest va Deep Learning (CNN/YOLO).
4. Soan noi dung slide bao cao 15-20 trang xuc tich, tap trung vao dong gop khoa hoc, ket qua thuc nghiem va ban demo.
Tieu chi nghiem thu: Quyen bao cao khoa hoc hoan chinh, so lieu trung thuc, he thong tai lieu tham khao chuan APA/IEEE.
Cap nhat checklist Giai doan 8 va tong ket du an.
[KET THUC PROMPT GIAI DOAN 8]

================================================================================
PHAN 9: BANG THEO DOI TIEN DO VA CHECKLIST TONG THE
================================================================================

Bang theo doi tien do thuc hien du an:
--------------------------------------------------------------------------------
Giai doan | Ten giai doan                 | Trang thai    | San pham nghiem thu
--------------------------------------------------------------------------------
GD 0      | Kiem ke & Chuan hoa moi truong| DANG THUC HIEN| test_env.py, Venv, KH NLN.md
GD 1      | Nghien cuu tong quan ly thuyet| CHUA THUC HIEN| bao_cao_tong_quan_khoa_hoc.md
GD 2      | Du lieu & Thiet ke thuc nghiem| CHUA THUC HIEN| Dataset, data.yaml, data_checker
GD 3      | Thiet ke kien truc pipeline   | CHUA THUC HIEN| config.yaml, src skeleton code
GD 4      | Xay dung baseline toi thieu   | CHUA THUC HIEN| run_baseline.py, annotated.mp4
GD 5      | Thuc nghiem & Toi uu mo hinh  | CHUA THUC HIEN| best.pt, benchmark_ml_vs_dl.py, log
GD 6      | Xu ly logic vi pham & Ca kho  | CHUA THUC HIEN| violation_logic.py, evidence/
GD 7      | Hoan thien demo duy nhat      | CHUA THUC HIEN| main_app.py (PyQt6/Streamlit)
GD 8      | Danh gia tong the & Bao cao   | CHUA THUC HIEN| Bao cao nien luan, Slide PPT
--------------------------------------------------------------------------------

Quy tac cap nhat:
- Sau moi lan thuc hien mot giai doan, cap nhat trang thai tu CHUA THUC HIEN sang DANG THUC HIEN hoac HOAN THANH.
- Ghi ro ly do neu co cong viec bi BI CHAN (BLOCKED).
- Chi chuyen trang thai sang HOAN THANH khi da co san pham thuc te duoc kiem chung bang ma nguon va file log.

================================================================================
PHAN 10: CAC QUYET DINH DA CHOT VA QUYET DINH CHO XAC NHAN
================================================================================

10.1. Cac quyet dinh da chot (Approved Decisions)
- Loai bo toan bo stack web da tang cồng kềnh (FastAPI, Redis, NestJS, MongoDB, MinIO, React, Docker).
- Xay dung 1 giao dien demo cuc bo duy nhat de trinh dien ket qua nghien cuu.
- Pipeline AI la trai tim cua de tai: tap trung vao Detection, Tracking va Spatio-Temporal Violation Logic.
- Bo do thi va huan luyen duoc thuc hien tren o dia D de tan dung 170 GB dung luong trong.

10.2. Cac quyet dinh cho xac nhan tu sinh vien (Pending Decisions)
- Quyet dinh 1: Lua chon cong nghe cho giao dien demo duy nhat:
  + Phuong an A (Khuyen nghi): Desktop GUI bang PyQt6 / PySide6 (Chay cuc bo, nhe, rat de dong goi va trinh chieu khong can trinh duyet).
  + Phuong an B: Streamlit / Gradio (Giao dien web nhe cuc bo chay tren trinh duyet 1 tab, truc quan nhung can chay server local).
- Quyet dinh 2: Pham vi phat hien bien so xe (LPR / PaddleOCR):
  + De nghi dua vao phan Mo rong (Optional) sau khi da hoan thien pipeline phat hien vuot den do cot loi, de khong lam phan tan thoi gian va de phong chat luong anh video tu xa khong du net de doc bien so.
- Quyet dinh 3: Nguon video thuc nghiem:
  + Sinh vien da co san video quay nut giao thong thuc te chua, hay can ho tro tap video giao thong cong khai?

================================================================================
KET THUC FILE KE HOACH
================================================================================
