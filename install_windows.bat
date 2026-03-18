@echo off
echo ============================================
echo BRAIN TUMOR 3D VISUALIZATION
echo Quick Setup Script for Windows
echo ============================================
echo.

echo [1/3] Installing Python dependencies...
pip install --user opencv-python mediapipe PyOpenGL pygame tensorflow numpy Pillow
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Verifying installation...
python -c "import cv2, mediapipe, OpenGL, pygame, tensorflow; print('All packages installed successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Package verification failed
    pause
    exit /b 1
)

echo.
echo [3/3] Setup complete!
echo.
echo ============================================
echo Ready to run!
echo.
echo To start the application:
echo   python brain_tumor_3d.py
echo.
echo Controls:
echo   1 Hand  = Rotate brain
echo   2 Hands = Zoom in/out
echo   ESC/Q   = Exit
echo ============================================
echo.
pause
