@echo off
REM Download Git LFS files (dataset + checkpoints) with checks.
setlocal EnableExtensions
cd /d "%~dp0\.."

echo === Git LFS pull (Windows) ===
echo Project: %CD%
echo.

where git >nul 2>&1
if errorlevel 1 (
  echo ERROR: git not found. Install Git for Windows.
  exit /b 1
)

git lfs version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Git LFS not installed.
  echo Download: https://git-lfs.com/
  exit /b 1
)

git lfs install
echo.

echo Fetching LFS objects from origin...
git lfs fetch origin --all
if errorlevel 1 (
  echo.
  echo FETCH FAILED. Common causes:
  echo   - no internet / firewall
  echo   - private repo: run "git fetch" first and sign in to GitHub
  echo   - wrong folder: clone again from GitHub
  exit /b 1
)

echo.
echo Checking out LFS files into working tree...
git lfs checkout
if errorlevel 1 exit /b 1

git lfs pull origin main 2>nul
git lfs pull 2>nul

echo.
call "%~dp0verify_lfs_files.bat"
exit /b %ERRORLEVEL%
