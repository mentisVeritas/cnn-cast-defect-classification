@echo off
REM One-time setup on Windows: venv + PyTorch (CUDA GPU) + dependencies
REM Requires: NVIDIA GPU + up-to-date driver (CUDA 12.x)
setlocal EnableExtensions
cd /d "%~dp0\.."

where py >nul 2>&1
if errorlevel 1 (
  echo Python launcher "py" not found. Install Python 3.11 from https://www.python.org/downloads/
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing PyTorch with CUDA 12.4 (GPU)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo.
echo Verify GPU:
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo.
echo Done. Activate before each session:
echo   .venv\Scripts\activate.bat
echo.
echo No NVIDIA GPU? Use CPU wheel instead:
echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.
exit /b 0
