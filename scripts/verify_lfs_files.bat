@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

set "OK=1"
set "MIN_ZIP_MB=50"

if not exist "data\data.zip" (
  echo [FAIL] Missing data\data.zip
  set "OK=0"
  goto :summary
)

for %%A in ("data\data.zip") do set "ZIP_BYTES=%%~zA"
set /a "ZIP_MB=ZIP_BYTES/1048576"
echo data\data.zip size: %ZIP_MB% MB ^(%ZIP_BYTES% bytes^)

if %ZIP_MB% LSS %MIN_ZIP_MB% (
  echo [FAIL] data\data.zip is too small - still an LFS pointer, not the real archive.
  echo        Expected about 65-85 MB. Run: scripts\lfs_pull_windows.bat
  set "OK=0"
) else (
  echo [OK] data\data.zip looks like a real zip.
)

if exist "outputs\models\best_model.pth" (
  for %%A in ("outputs\models\best_model.pth") do set "PTH_BYTES=%%~zA"
  set /a "PTH_MB=PTH_BYTES/1048576"
  echo outputs\models\best_model.pth: %PTH_MB% MB
  if %PTH_MB% LSS 1 (
    echo [WARN] checkpoint too small - run scripts\lfs_pull_windows.bat
    set "OK=0"
  ) else (
    echo [OK] checkpoint present.
  )
) else (
  echo [INFO] outputs\models\best_model.pth not found ^(optional^).
)

:summary
echo.
if "%OK%"=="1" (
  echo All LFS checks passed. Next: scripts\unpack_dataset.bat
  exit /b 0
)
echo LFS download incomplete. See README.md section "Git LFS on Windows".
exit /b 1
