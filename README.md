# OpenVINO Face Detection with Device Selection

Real-time face detection application using Intel OpenVINO with flexible CPU/NPU device selection and optimized performance.

## 🚀 Features

- **Device Selection**: Choose between CPU and NPU execution via command-line
- **Optimized Performance**: NPU ~11-12ms, CPU ~12ms inference time
- **Real-time Display**: Live camera feed with face detection bounding boxes
- **Performance Monitoring**: Real-time inference time overlay
- **Clean UI**: Professional video display with essential metrics

## 📦 System Requirements

- **Python**: 3.12+ (tested on Python 3.12.3)
- **Operating System**: Linux (Ubuntu/Debian recommended)
- **Hardware**: 
  - Webcam/Camera for video input
  - Intel NPU (optional, for NPU acceleration)
  - Intel CPU (fallback option)

## 🔧 Dependencies

**Core Dependencies:**
```
numpy==1.26.4           # Numerical computing
opencv-python==4.12.0.88   # Computer vision library  
openvino==2024.6.0      # Intel OpenVINO runtime
openvino-dev==2024.6.0  # OpenVINO development tools
```

**Supporting Dependencies:**
```
certifi==2025.8.3      # SSL certificates
charset-normalizer==3.4.3  # Character encoding
defusedxml==0.7.1       # Secure XML parsing
idna==3.10              # Internationalized domain names
networkx==3.1           # Network analysis
openvino-telemetry==2025.2.0  # OpenVINO telemetry
packaging==25.0         # Python packaging utilities
PyYAML==6.0.2          # YAML parser
requests==2.32.4       # HTTP library
urllib3==2.5.0         # HTTP client
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/wangdetash/OpenCV.git
cd OpenCV
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download OpenVINO Models
The application requires Intel face detection models. Download them from OpenVINO Model Zoo:

```bash
# Create model directory
mkdir -p intel/face-detection-adas-0001/FP16
mkdir -p intel/face-detection-adas-0001/FP16-INT8

# Download FP16 model (for CPU)
cd intel/face-detection-adas-0001/FP16
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-adas-0001/FP16/face-detection-adas-0001.bin

# Download FP16-INT8 model (for NPU)
cd ../FP16-INT8
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.bin

cd ../../..
```

## 🏃‍♂️ Usage

### Basic Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Run with NPU (default, best performance)
python3 open_camera_vision.py

# Run with CPU
python3 open_camera_vision.py --device CPU

# Run with NPU explicitly
python3 open_camera_vision.py --device NPU

# Show help
python3 open_camera_vision.py --help
```

### Expected Output
```
Loading FP16-INT8 quantized model for NPU
Model compiled for NPU device
640.0
480.0
640.0
480.0
NPU Inference time: 36.90 ms  # First inference (warmup)
NPU Inference time: 12.67 ms  # Steady state performance
NPU Inference time: 12.03 ms
...
```

### Controls
- **Press 'q'**: Quit the application
- **Video Display**: Shows live camera feed with face detection boxes
- **Performance Overlay**: Red text at top-right showing inference time and device

## 📊 Performance

| Device | Model | Inference Time | Use Case |
|--------|-------|----------------|----------|
| NPU    | FP16-INT8 | ~11-12ms | Best performance, low power |
| CPU    | FP16      | ~12ms    | Fallback, consistent performance |

## 📁 Project Structure
```
project/
├── open_camera_vision.py    # Main application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── intel/                  # OpenVINO models (excluded from git)
│   └── face-detection-adas-0001/
│       ├── FP16/           # CPU models
│       └── FP16-INT8/      # NPU models
├── cache/                  # OpenVINO cache (excluded from git)
└── venv/                   # Virtual environment (excluded from git)
```

## 🔍 Technical Details

### Device-Specific Optimizations
- **NPU Mode**: 
  - Uses FP16-INT8 quantized model for better performance
  - Dual DMA engines for parallel data transfer
  - Async inference requests for better hardware utilization
  - Optimized preprocessing with normalization

- **CPU Mode**: 
  - Uses standard FP16 precision model
  - Default OpenVINO CPU optimizations
  - Consistent performance across different hardware

### Video Processing
- **Input**: 640x480 webcam feed at 20fps
- **Detection**: Blue bounding boxes around detected faces
- **Confidence Threshold**: 0.5 (50%)
- **Output**: Real-time video display with performance overlay

## 🐛 Troubleshooting

### Common Issues

**1. Camera not found**
```bash
# Check camera devices
ls /dev/video*
# Ensure camera permissions
sudo chmod 666 /dev/video0
```

**2. NPU not available**
- Application automatically falls back to CPU
- Check Intel NPU driver installation

**3. Model files missing**
```bash
# Verify model files exist
ls -la intel/face-detection-adas-0001/FP16/
ls -la intel/face-detection-adas-0001/FP16-INT8/
```

**4. Permission issues**
```bash
# Fix camera permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### Verify Installation
```bash
python3 -c "import cv2, openvino; print('Dependencies OK')"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Intel OpenVINO Team for the optimization toolkit
- OpenCV community for computer vision libraries
- OpenVINO Model Zoo for pre-trained models

---

🤖 *Generated with [Claude Code](https://claude.ai/code)*