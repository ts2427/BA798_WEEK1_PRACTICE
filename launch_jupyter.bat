@echo off
REM Launch Jupyter Notebook for the ML Pipeline
REM Usage: launch_jupyter.bat

setlocal enabledelayedexpansion

echo.
echo ==============================================
echo ML Pipeline - Jupyter Launcher (Windows)
echo ==============================================
echo.

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo Error: pyproject.toml not found.
    echo Please run this script from the project root directory:
    echo   cd BA798_WEEK1_PRACTICE
    echo   launch_jupyter.bat
    pause
    exit /b 1
)

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.9+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Python found
echo [2/4] Installing Jupyter (if needed)...
python -m pip install -q jupyter ipykernel 2>nul || echo Jupyter installation skipped

echo [3/4] Starting Jupyter Notebook Server...
echo.
echo The notebook will open in your default browser.
echo If it doesn't, navigate to: http://localhost:8888
echo.
echo To open the pipeline notebook:
echo   1. Click on: notebooks/
echo   2. Click on: 05_main_pipeline.ipynb
echo.
echo To stop the server, close this window or press Ctrl+C
echo.
echo ==============================================
echo.

REM Start Jupyter
python -m jupyter notebook --notebook-dir=.

REM Keep window open if there's an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error occurred. Press any key to exit.
    pause
    exit /b 1
)

pause
