@echo off
REM Unpack data\data.zip after clone (Git LFS pulls the archive).
setlocal EnableExtensions
cd /d "%~dp0\.."

set "ZIP=%CD%\data\data.zip"
set "DEST=%CD%\data"

if exist "%DEST%\label.csv" if exist "%DEST%\raw_images\" (
  echo Already unpacked: %DEST%
  exit /b 0
)

if not exist "%ZIP%" (
  echo Missing %ZIP%
  echo Run: git lfs pull
  exit /b 1
)

if not exist "%DEST%" mkdir "%DEST%"
tar -xf "%ZIP%" -C "%DEST%"
if errorlevel 1 (
  echo Failed to unpack. Try: tar --version
  exit /b 1
)

echo Unpacked to %DEST% (label.csv + raw_images\)
exit /b 0
