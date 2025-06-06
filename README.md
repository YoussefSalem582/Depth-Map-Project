# 🎯 Depth Map Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-red.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modern web-based depth estimation and visualization platform built with Flask and OpenCV.

🌐 **Repository**: [https://github.com/YoussefSalem582/Depth-Map-Project](https://github.com/YoussefSalem582/Depth-Map-Project)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- VS Code (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YoussefSalem582/Depth-Map-Project.git
   cd Depth-Map-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   # Option 1: Simple launcher
   python run_app.py
   
   # Option 2: Direct Flask
   python app.py
   
   # Option 3: Platform scripts
   # Windows: run_flask.bat
   # Linux/Mac: ./run_flask.sh
   ```

4. **Open your browser** to `http://localhost:5000`

## 🌐 Web Interface Features

- **📱 Interactive Demo**: Generate and process synthetic depth maps
- **📷 Live Camera**: Real-time depth mapping from webcam feed
- **📤 Upload & Process**: Drag & drop image processing
- **📊 Metrics Analysis**: Comprehensive evaluation metrics
- **🎨 Colormap Gallery**: Visual depth map comparisons

## 🛠️ VS Code Setup

For full VS Code integration with debugging and tasks:

```bash
python setup_vscode.py
```

This creates:
- Virtual environment
- VS Code configurations
- Debug settings
- Automated tasks

## 📁 Project Structure

```
DepthMap/
├── 🌐 app.py                 # Flask web application
├── 🚀 run_app.py             # Simple launcher
├── 📋 requirements.txt       # Dependencies
├── 📁 src/depthmap/          # Core modules
│   ├── classical/            # Classical depth estimation
│   ├── eval/                 # Evaluation metrics
│   └── utils/                # Utilities & visualization
├── 📁 templates/             # HTML templates
└── 📁 uploads/               # File uploads
```

## 🎯 API Endpoints

- `GET /` - Main web interface
- `POST /api/demo` - Generate demo depth maps
- `POST /api/camera/start` - Start live camera feed
- `POST /api/camera/stop` - Stop camera feed
- `POST /api/camera/snapshot` - Capture camera snapshot
- `GET /api/camera/status` - Get camera status
- `GET /camera_feed` - Live camera stream
- `POST /api/upload` - Upload and process images
- `POST /api/evaluate` - Calculate evaluation metrics
- `GET /api/colormaps` - Get colormap examples

## 🔧 Development

### Core Features
- Depth map post-processing (hole filling, smoothing)
- Multiple visualization colormaps
- Comprehensive evaluation metrics
- Real-time web processing

### Adding New Features
1. Edit source code in `src/depthmap/`
2. Update Flask routes in `app.py`
3. Modify HTML template in `templates/index.html`

## 🐛 Troubleshooting

**Module not found errors:**
```bash
# Set PYTHONPATH (Windows)
set PYTHONPATH=%cd%\src

# Set PYTHONPATH (Linux/Mac)
export PYTHONPATH="$(pwd)/src"
```

**Dependencies missing:**
```bash
pip install -r requirements.txt
```

**Port already in use:**
- Change port in `app.py`: `app.run(port=5001)`

## 📊 Performance

- **Real-time processing**: Interactive depth map generation
- **Multiple formats**: PNG, JPG, JPEG, BMP, TIFF support
- **Responsive design**: Modern web interface
- **Cross-platform**: Windows, Linux, macOS

## 🎉 Success!

Your depth estimation web application is ready! Access it at `http://localhost:5000` and explore:

- 🔬 Depth estimation algorithms
- 📊 Performance metrics
- 🎨 Visualization options
- 📤 Image processing

**Happy coding!** 🚀 