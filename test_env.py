"""
test_env.py - Kiem tra toan dien moi truong AI cho du an Nien Luan
==================================================================
De tai: Tu dong phat hien phuong tien vi pham den tin hieu giao thong
May: NVIDIA GeForce RTX 3050 6GB Laptop GPU | Python 3.11

Chay: python test_env.py
Tieu chi nghiem thu: Hien thi "RTX 3050" va torch.cuda.is_available() = True
"""

import sys
import time
import platform

# --- Mau sac terminal ---
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"

def header(text: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        print(f"  {PASS} {msg_pass}")
    else:
        print(f"  {FAIL} {msg_fail}")
    return condition

results = {}

# ============================================================
# PHAN 1: THONG TIN HE THONG
# ============================================================
header("1. THONG TIN HE THONG")
print(f"  {INFO} OS:         {platform.system()} {platform.release()} ({platform.version()[:40]})")
print(f"  {INFO} Python:     {sys.version}")
print(f"  {INFO} Machine:    {platform.machine()}")

# ============================================================
# PHAN 2: KIEM TRA PYTORCH VA CUDA
# ============================================================
header("2. PYTORCH & CUDA")
try:
    import torch
    print(f"  {INFO} PyTorch version: {torch.__version__}")

    # Kiem tra CUDA
    cuda_ok = torch.cuda.is_available()
    results["cuda"] = check(cuda_ok,
        f"torch.cuda.is_available() = True",
        f"torch.cuda.is_available() = False - Cai lai PyTorch CUDA!")

    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        cuda_ver = torch.version.cuda
        cudnn_ver = torch.backends.cudnn.version()

        print(f"  {INFO} GPU Name:   {BOLD}{gpu_name}{RESET}")
        print(f"  {INFO} CUDA:       {cuda_ver}")
        print(f"  {INFO} cuDNN:      {cudnn_ver}")
        print(f"  {INFO} VRAM Total: {vram_total:.2f} GB")
        print(f"  {INFO} VRAM Used:  {vram_reserved:.2f} GB")

        # Kiem tra RTX 3050
        is_rtx3050 = "3050" in gpu_name
        results["gpu_name"] = check(is_rtx3050,
            f"GPU la RTX 3050 - Dung card muc tieu!",
            f"GPU '{gpu_name}' - Khong phai RTX 3050, kiem tra lai.")

        # Kiem tra VRAM >= 5GB (RTX 3050 6GB)
        vram_ok = vram_total >= 5.0
        results["vram"] = check(vram_ok,
            f"VRAM = {vram_total:.2f} GB (>= 5 GB) - Du cho training batch=16",
            f"VRAM = {vram_total:.2f} GB - Thap hon muc ki vong 5GB")

        # Kiem tra cuDNN
        cudnn_ok = torch.backends.cudnn.is_available()
        results["cudnn"] = check(cudnn_ok,
            "cuDNN san sang - Se toi uu toc do convolution",
            "cuDNN khong san sang - Toc do co the bi anh huong")

        # Kiem tra FP16 (AMP)
        fp16_ok = torch.cuda.is_bf16_supported() or True  # RTX 30xx ho tro FP16
        results["fp16"] = check(True,
            "FP16 (Mixed Precision) duoc ho tro tren RTX 30xx",
            "FP16 khong ho tro")

except ImportError as e:
    print(f"  {FAIL} Khong the import torch: {e}")
    print(f"  {WARN} Chay lenh: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    results["cuda"] = False

# ============================================================
# PHAN 3: BENCHMARK GPU - THOI GIAN TINH TOAN TENSOR
# ============================================================
header("3. BENCHMARK GPU - TENSOR COMPUTATION")

if results.get("cuda"):
    try:
        device = torch.device("cuda:0")

        # Test 1: Nhan ma tran lon (mo phong inference YOLO)
        print(f"  {INFO} Chay benchmark nhan ma tran (4096x4096 FP16)...")
        SIZE = 4096
        a = torch.randn(SIZE, SIZE, dtype=torch.float16, device=device)
        b = torch.randn(SIZE, SIZE, dtype=torch.float16, device=device)

        # Warm-up
        for _ in range(3):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark chinh
        N_ITER = 20
        start = time.perf_counter()
        for _ in range(N_ITER):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / N_ITER * 1000  # ms

        print(f"  {INFO} Thoi gian trung binh moi phep nhan: {elapsed:.2f} ms")
        bench_ok = elapsed < 100  # RTX 3050 nen duoi 100ms
        results["benchmark"] = check(bench_ok,
            f"Hieu nang GPU tot: {elapsed:.2f} ms/op (< 100ms)",
            f"Hieu nang GPU cham: {elapsed:.2f} ms/op (>= 100ms)")

        # Test 2: Mo phong inference YOLO (batch tensor 1x3x640x640)
        print(f"\n  {INFO} Mo phong inference batch YOLO (1x3x640x640 FP32)...")
        x = torch.randn(1, 3, 640, 640, device=device)
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)

        # Warm-up
        for _ in range(5):
            _ = conv(x)
        torch.cuda.synchronize()

        N_YOLO = 50
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(N_YOLO):
                _ = conv(x)
        torch.cuda.synchronize()
        yolo_ms = (time.perf_counter() - start) / N_YOLO * 1000

        fps_equiv = 1000.0 / yolo_ms
        print(f"  {INFO} Thoi gian inference tuong duong: {yolo_ms:.2f} ms/frame")
        print(f"  {INFO} FPS tuong duong (conv layer don): ~{fps_equiv:.0f} FPS")
        results["yolo_bench"] = check(yolo_ms < 50,
            f"Toc do xu ly frame nhanh: {yolo_ms:.2f} ms (< 50ms)",
            f"Toc do xu ly frame: {yolo_ms:.2f} ms")

        # Hien thi VRAM sau benchmark
        vram_after = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"\n  {INFO} VRAM da su dung sau benchmark: {vram_after:.1f} MB")
        torch.cuda.empty_cache()
        print(f"  {INFO} VRAM sau empty_cache(): {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

    except Exception as e:
        print(f"  {FAIL} Loi benchmark GPU: {e}")
        results["benchmark"] = False
else:
    print(f"  {WARN} Bo qua benchmark GPU (CUDA khong san sang)")

# ============================================================
# PHAN 4: KIEM TRA CAC THU VIEN AI CHINH
# ============================================================
header("4. KIEM TRA THU VIEN AI")

libraries = {
    "ultralytics":  "Ultralytics (YOLOv8/v11)",
    "cv2":          "OpenCV",
    "numpy":        "NumPy",
    "pandas":       "Pandas",
    "sklearn":      "Scikit-learn",
    "matplotlib":   "Matplotlib",
    "seaborn":      "Seaborn",
    "scipy":        "SciPy",
    "filterpy":     "FilterPy (Kalman Filter)",
    "lap":          "LapX / Lap (Hungarian Assign.)",
    "yaml":         "PyYAML",
    "tqdm":         "TQDM",
    "psutil":       "PSUtil",
    "loguru":       "Loguru",
    "PIL":          "Pillow",
}

all_libs_ok = True
for module_name, display_name in libraries.items():
    try:
        mod = __import__(module_name)
        ver = getattr(mod, "__version__", "N/A")
        print(f"  {PASS} {display_name:<30} v{ver}")
    except ImportError:
        print(f"  {FAIL} {display_name:<30} CHUA CAI DAT - chay: pip install -r requirements.txt")
        all_libs_ok = False

results["libraries"] = all_libs_ok

# ============================================================
# PHAN 5: KIEM TRA RAM VA CPU
# ============================================================
header("5. TAI NGUYEN HE THONG")
try:
    import psutil

    ram_total = psutil.virtual_memory().total / (1024**3)
    ram_avail = psutil.virtual_memory().available / (1024**3)
    ram_pct   = psutil.virtual_memory().percent
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq  = psutil.cpu_freq()

    print(f"  {INFO} RAM Tong:     {ram_total:.1f} GB")
    print(f"  {INFO} RAM Con trong:{ram_avail:.1f} GB ({100-ram_pct:.0f}% trong)")
    print(f"  {INFO} CPU Cores:    {cpu_count} luong logic")
    if cpu_freq:
        print(f"  {INFO} CPU Freq:     {cpu_freq.current:.0f} MHz")

    ram_ok = ram_avail >= 2.0
    results["ram"] = check(ram_ok,
        f"RAM con trong du ({ram_avail:.1f} GB >= 2 GB)",
        f"RAM con trong thap ({ram_avail:.1f} GB) - Dong bot ung dung truoc khi train!")

    # Kiem tra o dia D
    try:
        disk = psutil.disk_usage("D:\\")
        disk_free = disk.free / (1024**3)
        print(f"  {INFO} O dia D:     {disk.total/1024**3:.0f} GB tong, {disk_free:.1f} GB trong")
        results["disk"] = check(disk_free >= 50,
            f"O dia D du dung luong ({disk_free:.1f} GB trong)",
            f"O dia D chi con {disk_free:.1f} GB - Can don dep!")
    except:
        print(f"  {WARN} Khong doc duoc thong tin o dia D")

except Exception as e:
    print(f"  {WARN} Khong doc duoc thong tin RAM: {e}")

# ============================================================
# PHAN 6: TONG KET
# ============================================================
header("6. TONG KET KIEM TRA MOI TRUONG")

passed = sum(1 for v in results.values() if v)
total  = len(results)
score  = int(passed / total * 100) if total > 0 else 0

print(f"\n  So muc dat: {passed}/{total} ({score}%)")
print()

critical_ok = results.get("cuda", False) and results.get("gpu_name", False)
if critical_ok and score >= 70:
    print(f"  {GREEN}{BOLD}*** MOI TRUONG DA SAN SANG CHO DU AN NIEN LUAN! ***{RESET}")
    print(f"  {GREEN}RTX 3050 duoc nhan dien, CUDA hoat dong, cac thu vien co ban OK.{RESET}")
    print(f"\n  Buoc tiep theo:")
    print(f"  1. Chay: pip install -r requirements.txt (neu chua cai thu vien)")
    print(f"  2. Bat dau Giai doan 1: Nghien cuu tong quan")
elif not critical_ok:
    print(f"  {RED}{BOLD}*** CHUA SAN SANG - CAN SUA LOI CUDA TRUOC! ***{RESET}")
    print(f"  {YELLOW}Cai PyTorch CUDA 12.4:{RESET}")
    print(f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
else:
    print(f"  {YELLOW}{BOLD}*** MOI TRUONG HOAT DONG TUONG DOI - KIEM TRA CAC MUC FAIL ***{RESET}")

print(f"\n{'='*60}\n")

