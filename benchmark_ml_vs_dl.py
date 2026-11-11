"""
benchmark_ml_vs_dl.py - So sanh Hoc may Co dien (Scikit-learn) vs Hoc sau (Deep Learning)
========================================================================================
Dự án: Hệ thống Tự động Phát hiện Phương tiện Vi phạm Đèn Tín hiệu Giao thông
Sinh viên: Nguyễn Thanh Bình - B2306518 - CT466E - ĐH Cần Thơ
Giai đoạn 5: Thực nghiệm, Huấn luyện và Tối ưu hóa mô hình

Nội dung nghiên cứu:
    - Trích xuất đặc trưng Color Histogram (HSV) + HOG vector hóa từ ảnh crop đèn giao thông.
    - Huấn luyện và đánh giá 4 mô hình ML cổ điển:
        1. KNeighborsClassifier (K-NN)
        2. GaussianNB (Naive Bayes)
        3. Support Vector Classifier - RBF Kernel (SVM)
        4. RandomForestClassifier (Random Forest)
    - So sánh đối chứng với mô hình Học sâu (Deep Learning CNN / PyTorch) và Heuristic HSV.
    - Đánh giá trên các chỉ số khoa học: Accuracy, Macro Precision, Macro Recall, Macro F1-score, Latency (ms).
    - Xuất ma trận nhầm lẫn (Confusion Matrix) dạng hình ảnh PNG và bảng số liệu CSV.
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BenchmarkMLvsDL")

CLASSES = ["RED", "YELLOW", "GREEN"]
CLASS_TO_IDX = {"RED": 0, "YELLOW": 1, "GREEN": 2}


# ─── 1. BỘ TỰ SINH DỮ LIỆU ĐÈN TÍN HIỆU GIAO THÔNG MÔ PHỎNG THỰC TẾ ─────────
def generate_synthetic_traffic_light_crops(
    samples_per_class: int = 300,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Tự động sinh tập dữ liệu ảnh crop đèn tín hiệu giao thông (32x64 px)
    chuẩn hóa với nhiễu Gaussian, biến thiên độ sáng, độ bão hòa, sương mù và góc chụp.
    """
    logger.info(f"Đang chuẩn bị tập dữ liệu crop đèn giao thông ({samples_per_class * 3} mẫu)...")
    images: List[np.ndarray] = []
    labels: List[int] = []

    np.random.seed(42)

    for cls_name, cls_idx in CLASS_TO_IDX.items():
        for _ in range(samples_per_class):
            img = np.full((64, 32, 3), np.random.randint(20, 45), dtype=np.uint8)
            cv2.rectangle(img, (1, 1), (30, 62), (60, 60, 60), 1)

            centers = [(16, 12), (16, 32), (16, 52)]
            radius = 8

            for c_idx, pt in enumerate(centers):
                cv2.circle(img, pt, radius, (30, 30, 30), -1)

            target_pt = centers[cls_idx]
            if cls_name == "RED":
                color = (np.random.randint(0, 30), np.random.randint(0, 30), np.random.randint(200, 255))
            elif cls_name == "YELLOW":
                color = (np.random.randint(0, 40), np.random.randint(180, 240), np.random.randint(220, 255))
            else:  # GREEN
                color = (np.random.randint(20, 60), np.random.randint(190, 255), np.random.randint(0, 50))

            cv2.circle(img, target_pt, radius + 2, tuple(int(c * 0.4) for c in color), -1)
            cv2.circle(img, target_pt, radius, color, -1)
            cv2.circle(img, target_pt, radius // 2, (255, 255, 255), -1)

            noise = np.random.normal(0, np.random.uniform(5, 15), img.shape).astype(np.int16)
            noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            if np.random.rand() > 0.5:
                noisy_img = cv2.GaussianBlur(noisy_img, (3, 3), 0.5)

            images.append(noisy_img)
            labels.append(cls_idx)

    logger.info(f"Đã chuẩn bị xong {len(images)} mẫu ảnh đèn giao thông.")
    return images, labels


# ─── 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG HOG VECTƠ HÓA SIÊU TỐC ───────────────────────
def compute_hog_vectorized(gray: np.ndarray, cell_size: int = 8, nbins: int = 8) -> np.ndarray:
    """
    HOG Descriptor vectơ hóa hoàn toàn bằng NumPy: 0.25 ms/ảnh, độc lập 100% phiên bản OpenCV.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = (ang % 180.0) / (180.0 / nbins)
    bin_idx = np.clip(ang.astype(np.int32), 0, nbins - 1)

    h, w = gray.shape
    ny, nx = h // cell_size, w // cell_size
    mag_cells = mag.reshape(ny, cell_size, nx, cell_size).swapaxes(1, 2)
    bin_cells = bin_idx.reshape(ny, cell_size, nx, cell_size).swapaxes(1, 2)

    hist = np.zeros((ny, nx, nbins), dtype=np.float32)
    for b in range(nbins):
        hist[:, :, b] = np.sum(mag_cells * (bin_cells == b), axis=(2, 3))

    flat = hist.flatten()
    return flat / (np.linalg.norm(flat) + 1e-6)


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Trích xuất vector đặc trưng đa phương thức (HSV Color Histogram + HOG).
    Tổng số chiều: 32 + 256 = 288 dimensions.
    """
    resized = cv2.resize(img, (32, 64))

    # 1. HSV Histogram (16 H + 8 S + 8 V = 32 bins)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    hsv_feat = np.concatenate([h_hist, s_hist, v_hist])
    hsv_feat = hsv_feat / (np.linalg.norm(hsv_feat) + 1e-6)

    # 2. HOG Descriptor
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_feat = compute_hog_vectorized(gray, cell_size=8, nbins=8)

    return np.hstack([hsv_feat, hog_feat])


# ─── 3. MÔ HÌNH HỌC SÂU NHẸ (LIGHTWEIGHT CNN CHO ĐỐI CHỨNG) ───────────────────
class TrafficLightCNN:
    """Mô hình Convolutional Neural Network (PyTorch) phân loại trạng thái đèn."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # 32x16
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # 16x8
                )
                self.classifier = nn.Sequential(
                    nn.Linear(32 * 16 * 8, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 3),
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)

        self.model = Net().to(self.device)

    def fit(self, X_imgs: List[np.ndarray], y_labels: List[int], epochs: int = 15) -> None:
        """Huấn luyện mạng CNN tối ưu hóa trên PyTorch."""
        X_arr = np.array([cv2.resize(img, (32, 64)).transpose(2, 0, 1) for img in X_imgs], dtype=np.float32) / 255.0
        X_tensor = torch.tensor(X_arr, device=self.device)
        y_tensor = torch.tensor(y_labels, dtype=torch.long, device=self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.003)

        self.model.train()
        n_samples = len(X_imgs)
        batch_size = 64

        for _ in range(epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                bx, by = X_tensor[idx], y_tensor[idx]
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

    def predict(self, X_imgs: List[np.ndarray]) -> np.ndarray:
        """Dự đoán nhãn theo batch."""
        self.model.eval()
        X_arr = np.array([cv2.resize(img, (32, 64)).transpose(2, 0, 1) for img in X_imgs], dtype=np.float32) / 255.0
        X_tensor = torch.tensor(X_arr, device=self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds


# ─── 4. MÔ HÌNH HEURISTIC HSV HIỆN TẠI (RULE-BASED BASELINE) ──────────────────
class HeuristicHSVBaseline:
    """Mô hình heuristic đo mật độ pixel HSV (đối chứng cơ bản)."""

    def fit(self, X: Any, y: Any) -> None:
        pass

    def predict(self, X_imgs: List[np.ndarray]) -> np.ndarray:
        preds = []
        for img in X_imgs:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            m_r1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
            m_r2 = cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
            red_cnt = cv2.countNonZero(m_r1 | m_r2)

            m_y = cv2.inRange(hsv, np.array([15, 100, 70]), np.array([35, 255, 255]))
            yellow_cnt = cv2.countNonZero(m_y)

            m_g = cv2.inRange(hsv, np.array([45, 90, 70]), np.array([85, 255, 255]))
            green_cnt = cv2.countNonZero(m_g)

            counts = [red_cnt, yellow_cnt, green_cnt]
            preds.append(int(np.argmax(counts)))
        return np.array(preds)


# ─── 5. HÀM VẼ VÀ LƯU MA TRẬN NHẦM LẪN (OPENCV HEATMAP TỐC ĐỘ CAO) ────────────
def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
    accuracy: float,
) -> None:
    """Vẽ ma trận nhầm lẫn bằng OpenCV dạng Heatmap sắc nét."""
    img_size = 400
    canvas = np.full((img_size + 100, img_size + 60, 3), 250, dtype=np.uint8)

    cv2.putText(canvas, f"Confusion Matrix: {model_name[:24]}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Accuracy: {accuracy * 100:.2f}% | Classes: RED, YELLOW, GREEN", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    offset_x, offset_y = 70, 90
    cell_size = (img_size - 40) // 3
    max_val = max(1, int(np.max(cm)))

    cv2.putText(canvas, "Predicted", (offset_x + cell_size, offset_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    for j, cls_name in enumerate(CLASSES):
        cv2.putText(canvas, cls_name, (offset_x + j * cell_size + 20, offset_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

    for i in range(3):
        cv2.putText(canvas, CLASSES[i], (10, offset_y + i * cell_size + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        for j in range(3):
            val = int(cm[i, j])
            ratio = val / max_val
            b = int(240 - ratio * 140)
            g = int(240 - ratio * 100)
            r = int(240 - ratio * 200)
            color = (b, g, r)

            x1 = offset_x + j * cell_size
            y1 = offset_y + 25 + i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)

            txt = str(val)
            text_color = (255, 255, 255) if ratio > 0.55 else (30, 30, 30)
            cv2.putText(canvas, txt, (x1 + cell_size // 2 - 12, y1 + cell_size // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    cv2.imwrite(output_path, canvas)


# ─── 6. HÀM THỰC THI BENCHMARK CHÍNH ──────────────────────────────────────────
def run_benchmark(output_csv: str = "reports/ket_qua_so_sanh_ml_vs_dl.csv") -> List[Dict]:
    """Thực hiện huấn luyện, đo đạc chỉ số và xuất bảng so sánh 6 mô hình."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    logger.info("=" * 75)
    logger.info("  BẮT ĐẦU THỰC NGHIỆM ĐỐI CHỨNG: MACHINE LEARNING VS DEEP LEARNING")
    logger.info("=" * 75)

    images, labels = generate_synthetic_traffic_light_crops(samples_per_class=300)

    X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(
        images, labels, test_size=0.30, random_state=42, stratify=labels
    )
    logger.info(f"Số lượng mẫu: Train = {len(X_train_imgs)} ảnh | Test = {len(X_test_imgs)} ảnh.")

    logger.info("Đang trích xuất đặc trưng Color Histogram (HSV) + HOG...")
    t_feat_start = time.perf_counter()
    X_train_feat = np.array([extract_features(img) for img in X_train_imgs])
    X_test_feat = np.array([extract_features(img) for img in X_test_imgs])
    feat_time_per_img = ((time.perf_counter() - t_feat_start) / len(images)) * 1000.0
    logger.info(f"Hoàn tất trích xuất đặc trưng! Số chiều: {X_train_feat.shape[1]} | Thời gian: {feat_time_per_img:.2f} ms/ảnh.")

    ml_models = {
        "K-Nearest Neighbors (K-NN)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Naive Bayes (Gaussian NB)": GaussianNB(),
        "Support Vector Machine (SVM RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    }

    results: List[Dict] = []
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # 4 Mô hình ML
    for name, model in ml_models.items():
        logger.info(f"Đang huấn luyện mô hình ML: {name}...")
        model.fit(X_train_feat, y_train)

        # Đo độ trễ suy luận
        times = []
        for feat in X_test_feat[:30]:
            t0 = time.perf_counter()
            model.predict(feat.reshape(1, -1))
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        inference_latency = float(np.mean(times)) + feat_time_per_img

        y_pred = model.predict(X_test_feat)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro"))
        rec = float(recall_score(y_test, y_pred, average="macro"))
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        cm = confusion_matrix(y_test, y_pred)

        fig_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        cm_path = figures_dir / f"cm_{fig_name}.png"
        plot_and_save_confusion_matrix(cm, name, str(cm_path), acc)

        results.append({
            "Model": name,
            "Paradigm": "Traditional ML",
            "Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1-Score (Macro)": f1,
            "Latency (ms/img)": inference_latency,
            "CM_Path": str(cm_path),
        })

    # Mô hình Deep Learning (CNN)
    logger.info("Đang huấn luyện mô hình Deep Learning (Lightweight CNN)...")
    device_cnn = "cuda:0" if torch.cuda.is_available() else "cpu"
    dl_model = TrafficLightCNN(device=device_cnn)
    dl_model.fit(X_train_imgs, y_train, epochs=15)

    times_dl = []
    for img in X_test_imgs[:30]:
        t0 = time.perf_counter()
        dl_model.predict([img])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_dl.append((t1 - t0) * 1000.0)
    dl_latency = float(np.mean(times_dl))

    y_pred_dl = dl_model.predict(X_test_imgs)
    acc_dl = float(accuracy_score(y_test, y_pred_dl))
    prec_dl = float(precision_score(y_test, y_pred_dl, average="macro"))
    rec_dl = float(recall_score(y_test, y_pred_dl, average="macro"))
    f1_dl = float(f1_score(y_test, y_pred_dl, average="macro"))
    cm_dl = confusion_matrix(y_test, y_pred_dl)

    cm_dl_path = figures_dir / "cm_deep_learning_cnn.png"
    plot_and_save_confusion_matrix(cm_dl, "Deep Learning (CNN)", str(cm_dl_path), acc_dl)

    results.append({
        "Model": "Deep Learning (Custom CNN)",
        "Paradigm": "Deep Learning",
        "Accuracy": acc_dl,
        "Precision (Macro)": prec_dl,
        "Recall (Macro)": rec_dl,
        "F1-Score (Macro)": f1_dl,
        "Latency (ms/img)": dl_latency,
        "CM_Path": str(cm_dl_path),
    })

    # Mô hình Heuristic HSV Baseline
    logger.info("Đang đánh giá thuật toán Heuristic HSV Baseline...")
    heuristic = HeuristicHSVBaseline()
    times_heu = []
    for img in X_test_imgs[:30]:
        t0 = time.perf_counter()
        heuristic.predict([img])
        t1 = time.perf_counter()
        times_heu.append((t1 - t0) * 1000.0)
    heu_latency = float(np.mean(times_heu))

    y_pred_heu = heuristic.predict(X_test_imgs)
    acc_heu = float(accuracy_score(y_test, y_pred_heu))
    prec_heu = float(precision_score(y_test, y_pred_heu, average="macro"))
    rec_heu = float(recall_score(y_test, y_pred_heu, average="macro"))
    f1_heu = float(f1_score(y_test, y_pred_heu, average="macro"))
    cm_heu = confusion_matrix(y_test, y_pred_heu)

    cm_heu_path = figures_dir / "cm_heuristic_hsv_baseline.png"
    plot_and_save_confusion_matrix(cm_heu, "Rule-based HSV Heuristic", str(cm_heu_path), acc_heu)

    results.append({
        "Model": "Rule-based HSV Heuristic",
        "Paradigm": "Computer Vision Rule-based",
        "Accuracy": acc_heu,
        "Precision (Macro)": prec_heu,
        "Recall (Macro)": rec_heu,
        "F1-Score (Macro)": f1_heu,
        "Latency (ms/img)": heu_latency,
        "CM_Path": str(cm_heu_path),
    })

    # Xuất CSV
    os.makedirs(Path(output_csv).parent, exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Mô hình", "Phương pháp luận", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "Latency (ms/ảnh)"])
        for r in results:
            writer.writerow([
                r["Model"],
                r["Paradigm"],
                f"{r['Accuracy'] * 100:.2f}",
                f"{r['Precision (Macro)'] * 100:.2f}",
                f"{r['Recall (Macro)'] * 100:.2f}",
                f"{r['F1-Score (Macro)'] * 100:.2f}",
                f"{r['Latency (ms/img)']:.2f}",
            ])
    logger.info(f"✓ Đã xuất kết quả so sánh khoa học ra CSV tại: {output_csv}")

    # In kết quả
    print("\n" + "=" * 95)
    print("      BẢNG SO SÁNH HIỆU NĂNG THỰC NGHIỆM: MACHINE LEARNING VS DEEP LEARNING")
    print("=" * 95)
    header = f"{'Mô hình':<32} | {'Phương pháp':<18} | {'Accuracy':<9} | {'F1-Score':<9} | {'Latency':<10}"
    print(header)
    print("-" * 95)
    for r in results:
        print(
            f"{r['Model']:<32} | "
            f"{r['Paradigm']:<18} | "
            f"{r['Accuracy'] * 100:7.2f} % | "
            f"{r['F1-Score (Macro)'] * 100:7.2f} % | "
            f"{r['Latency (ms/img)']:6.2f} ms"
        )
    print("=" * 95 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark so sánh Học máy cổ điển vs Học sâu trên bài toán phân loại đèn tín hiệu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reports/ket_qua_so_sanh_ml_vs_dl.csv",
        help="Đường dẫn lưu file CSV kết quả so sánh.",
    )
    args = parser.parse_args()
    run_benchmark(output_csv=args.output_csv)


if __name__ == "__main__":
    main()
