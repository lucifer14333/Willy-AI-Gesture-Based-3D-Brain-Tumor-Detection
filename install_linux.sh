#!/bin/bash

echo "============================================"
echo "BRAIN TUMOR 3D VISUALIZATION"
echo "Quick Setup Script for Linux"
echo "============================================"
echo ""

echo "[1/3] Installing Python dependencies..."
pip3 install --break-system-packages opencv-python mediapipe PyOpenGL pygame tensorflow numpy Pillow || {
    echo "ERROR: Failed to install dependencies"
    exit 1
}

echo ""
echo "[2/3] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y freeglut3-dev || {
    echo "WARNING: Could not install GLUT (may already be installed)"
}

echo ""
echo "[3/3] Verifying installation..."
python3 -c "import cv2, mediapipe, OpenGL, pygame, tensorflow; print('All packages installed successfully!')" || {
    echo "ERROR: Package verification failed"
    exit 1
}

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To start the application:"
echo "  python3 brain_tumor_3d.py"
echo ""
echo "Controls:"
echo "  1 Hand  = Rotate brain"
echo "  2 Hands = Zoom in/out"
echo "  ESC/Q   = Exit"
echo "============================================"
