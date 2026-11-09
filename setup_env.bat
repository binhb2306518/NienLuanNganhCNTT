@echo off
REM ================================================================
REM setup_env.bat - Cai dat moi truong tu dong cho du an Nien Luan
REM May: RTX 3050 6GB | CUDA 12.4 | Python 3.11
REM ================================================================
REM Chay bang cach: Right-click -> Run as Administrator (neu can)
REM hoac mo PowerShell tai D:\NIENLUANNGANH va chay: .\setup_env.bat
REM ================================================================

echo.
echo ================================================================
echo   THIET LAP MOI TRUONG - DU AN NIEN LUAN AI
echo   RTX 3050 6GB + CUDA 12.4 + Python 3.11
echo ================================================================
echo.

REM --- Buoc 1: Tao moi truong ao ---
echo [1/4] Tao virtual environment tai D:\NIENLUANNGANH\venv_nln ...
python -m venv D:\NIENLUANNGANH\venv_nln
if errorlevel 1 (
    echo [LOI] Khong tao duoc venv. Kiem tra Python 3.11 da cai chua.
    pause
    exit /b 1
)
echo [OK] Virtual environment da duoc tao.
echo.

REM --- Buoc 2: Kich hoat venv va nang cap pip ---
echo [2/4] Kich hoat venv va nang cap pip...
call D:\NIENLUANNGANH\venv_nln\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
echo [OK] pip da duoc nang cap.
echo.

REM --- Buoc 3: Cai dat PyTorch CUDA 12.4 ---
echo [3/4] Cai dat PyTorch 2.4+ voi ho tro CUDA 12.4...
echo (Qua trinh nay co the mat 5-15 phut tuy toc do internet)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [LOI] Cai dat PyTorch that bai. Kiem tra ket noi internet.
    pause
    exit /b 1
)
echo [OK] PyTorch CUDA da cai dat.
echo.

REM --- Buoc 4: Cai dat cac thu vien con lai ---
echo [4/4] Cai dat cac thu vien AI tu requirements.txt...
pip install -r D:\NIENLUANNGANH\requirements.txt
if errorlevel 1 (
    echo [CANH BAO] Mot so thu vien cai that bai. Kiem tra log o tren.
)
echo.

REM --- Kiem tra ket qua ---
echo ================================================================
echo   KIEM TRA MOI TRUONG...
echo ================================================================
python D:\NIENLUANNGANH\test_env.py

echo.
echo ================================================================
echo   CAI DAT HOAN TAT!
echo   De kich hoat moi truong trong tuong lai, chay:
echo     D:\NIENLUANNGANH\venv_nln\Scripts\activate.bat
echo ================================================================
pause

