# 🧠 Willy-AI: Gesture-Based 3D Brain Tumor Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenGL](https://img.shields.io/badge/OpenGL-3.1.7-green.svg)](https://www.opengl.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.8-red.svg)](https://mediapipe.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)

> **A touchless, gesture-controlled 3D brain visualization system for medical education and tumor detection using AI and computer vision.**

![Willy-AI Demo](assets/demo-banner.png)

## 🎯 Overview

**Willy-AI** is an innovative medical imaging tool that combines gesture recognition, deep learning, and 3D visualization to create an intuitive, touchless interface for exploring brain anatomy and detecting tumors. Built for medical students, educators, and healthcare professionals.

### ✨ Key Features

- 🖐️ **Touchless Gesture Control** - Navigate 3D brain models using natural hand gestures via webcam
- 🧠 **AI-Powered Tumor Detection** - 95% accuracy using lightweight MobileNetV2 architecture (2.8 MB)
- 🎨 **Real-time 3D Visualization** - Smooth 60 FPS rendering with OpenGL
- 🔬 **X-Ray Mode** - Toggle brain transparency to view internal tumor locations
- 📊 **Professional UI** - Elegant interface with live detection metrics and confidence scores
- 💾 **MRI Upload** - Analyze custom brain MRI scans in real-time

## 🎥 Demo

### Gesture Controls

| Gesture | Action | Demo |
|---------|--------|------|
| 👊 **Closed Fist** | X-Ray Mode (20% transparency) | ![Fist](assets/gesture-fist.gif) |
| 🖐️ **Open Hand** | 360° Rotation | ![Rotate](assets/gesture-rotate.gif) |
| ✌️ **Two Hands** | Zoom In/Out | ![Zoom](assets/gesture-zoom.gif) |

### System in Action

![Detection Demo](assets/detection-demo.gif)

*Real-time tumor detection and 3D visualization*

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Webcam
- Windows/Linux/macOS

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/willy-ai-brain-tumor.git
cd willy-ai-brain-tumor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Required Files

1. **3D Brain Model** (42 MB): [Download brain_model.glb](https://drive.google.com/your-link)
2. **AI Model** (2.8 MB): [Download brain_tumor_model_lite.h5](https://drive.google.com/your-link)
3. **Sample Dataset**: [Download brain_mri_dataset.zip](https://drive.google.com/your-link)

Place files in the project root:
```
willy-ai-brain-tumor/
├── brain_model.glb
├── brain_tumor_model_lite.h5
└── brain_mri_dataset/
    ├── yes/
    └── no/
```

### Run the Application

```bash
python brain_tumor_3d_final.py
```

**Optional Enhanced UI:**
```bash
# Download ui_enhancements.py for fancy UI
python brain_tumor_3d_final.py  # Auto-detects and loads UI
```

## 📖 Usage

### Controls

- **👊 Fist** - Activate X-Ray Mode (brain becomes transparent)
- **🖐️ Open Hand** - Rotate brain 360° (index finger controls direction)
- **✌️ Two Hands** - Zoom in/out (distance between hands controls zoom level)
- **U Key** - Upload custom MRI scan
- **ESC** - Exit application
- **Mouse Drag** - Manual rotation
- **Scroll Wheel** - Manual zoom

### Uploading MRI Scans

1. Press `U` key
2. Select MRI image (JPEG/PNG)
3. AI analyzes in real-time
4. View results with confidence score
5. Tumor visualized in 3D if detected

## 🏗️ Architecture

### System Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Webcam    │───▶│   MediaPipe  │───▶│   Gesture   │
│   Input     │    │  Hand Track  │    │  Recognition│
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  3D Brain   │◀───│    OpenGL    │◀───│  Transform  │
│   Model     │    │   Renderer   │    │   Engine    │
└─────────────┘    └──────────────┘    └─────────────┘
                          ▲
                          │
┌─────────────┐    ┌──────────────┐
│  MRI Scan   │───▶│  MobileNetV2 │
│   Upload    │    │  AI Model    │
└─────────────┘    └──────────────┘
```

### AI Model Architecture

**MobileNetV2-based CNN:**
- Input: 128×128×3 MRI images
- Backbone: Pretrained MobileNetV2 (53 layers)
- Global Average Pooling: 4×4×1280 → 1280
- Dense Layer: 64 neurons (ReLU activation)
- Dropout: 30% regularization
- Output: Sigmoid (0-100% confidence)

**Performance:**
- Accuracy: 95.1%
- Model Size: 2.8 MB
- Inference Time: <50ms

## 📊 Dataset

**Source:** [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

**Statistics:**
- Total Images: 253
- Tumor (Yes): 155 images
- Normal (No): 98 images
- Split: 80% Training (203) | 20% Validation (50)

## 🛠️ Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Core programming language | 3.10.11 |
| MediaPipe | Hand tracking & gesture recognition | 0.10.8 |
| OpenGL/PyOpenGL | 3D rendering engine | 3.1.7 |
| Pygame | Window management & UI | 2.5.2 |
| OpenCV | Camera capture & image processing | 4.8.1.78 |
| pygltflib | 3D model loading (GLB format) | Latest |
| TensorFlow/Keras | Deep learning framework | 2.15.0 |
| NumPy | Array operations & mathematics | 1.24.3 |
| Pillow (PIL) | Image loading & processing | 10.1.0 |

## 📁 Project Structure

```
willy-ai-brain-tumor/
│
├── brain_tumor_3d_final.py      # Main application
├── ui_enhancements.py            # Optional enhanced UI module
├── brain_model.glb               # 3D brain model (901,839 vertices)
├── brain_tumor_model_lite.h5     # Trained AI model (2.8 MB)
│
├── brain_mri_dataset/            # Training dataset
│   ├── yes/                      # Tumor-positive MRI scans
│   └── no/                       # Normal MRI scans
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
│
├── docs/                         # Documentation
│   ├── INSTALLATION.md
│   ├── USER_GUIDE.md
│   └── API_REFERENCE.md
│
├── assets/                       # Media assets
│   ├── demo-banner.png
│   ├── gesture-fist.gif
│   ├── gesture-rotate.gif
│   └── gesture-zoom.gif
│
└── presentation/                 # Project presentation
    └── Willy-AI_Presentation.pptx
```

## 🎨 Features in Detail

### 1. Gesture Recognition (MediaPipe)

- **21 hand landmarks** tracked per hand in real-time
- **Sub-pixel accuracy** for precise gesture detection
- **Multi-hand support** for two-handed gestures (zoom)
- **Fist detection algorithm**: Measures average distance from palm to fingertips (<0.12 threshold)

### 2. 3D Visualization

- **High-poly brain model**: 901,839 vertices for anatomical accuracy
- **Tumor markers**: Red pulsating spheres with glow effects and animated rings
- **Transparency modes**: Normal (75% brain) vs X-Ray (20% brain)
- **Smooth rendering**: 60 FPS with depth testing and alpha blending

### 3. AI Detection

- **Lightweight model**: MobileNetV2 architecture (2.8 MB)
- **Fast inference**: Real-time analysis (<50ms per scan)
- **Confidence scoring**: 0-100% probability with color-coded indicators
- **Batch processing**: Can analyze multiple scans sequentially

### 4. Professional UI

- **Top bar**: FPS counter, gesture mode indicator
- **Detection panel**: Status, confidence bar, AI model info
- **Tumor status card**: Large percentage display with pulsing indicator
- **Camera feed**: Live gesture tracking visualization
- **MRI display**: Uploaded scan preview
- **Instructions**: Context-aware control hints

## 🔬 Technical Highlights

### Gesture Detection Algorithm

```python
def _is_fist(hand_landmarks):
    """Detect closed fist gesture"""
    palm = hand_landmarks[0]
    fingertips = [hand_landmarks[i] for i in [4, 8, 12, 16, 20]]
    
    distances = [
        sqrt((tip.x - palm.x)**2 + (tip.y - palm.y)**2)
        for tip in fingertips
    ]
    
    avg_distance = sum(distances) / len(distances)
    return avg_distance < 0.12  # Threshold for closed fist
```

### Tumor Visualization

```python
# Multi-layered tumor rendering
1. Glow sphere (outer layer, 2× size, 40% opacity)
2. Core sphere (inner layer, 1× size, 95% opacity)
3. Rotating wire ring (animated, orange)
4. Depth-independent rendering (always visible)
```

## 📈 Performance

- **FPS**: 60 (stable)
- **Gesture Recognition Accuracy**: 95%+
- **AI Detection Accuracy**: 95.1%
- **Model Size**: 2.8 MB
- **RAM Usage**: ~500 MB
- **GPU**: Optional (CPU-only supported)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black brain_tumor_3d_final.py
flake8 brain_tumor_3d_final.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **3D Brain Model**: University of Dundee
- **Dataset**: [Navoneel Chakrabarty](https://www.kaggle.com/navoneel) (Kaggle)
- **MediaPipe**: Google Research
- **OpenGL Community**: For rendering pipeline insights

## 📧 Contact

**Aaron Leob**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

## 🌟 Support

If you found this project helpful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs
- 💡 Suggesting new features
- 📣 Sharing with others

---

<div align="center">
  <p>Made with ❤️ for medical education and innovation</p>
  <p>
    <a href="#-willy-ai-gesture-based-3d-brain-tumor-detection">Back to Top ↑</a>
  </p>
</div>
