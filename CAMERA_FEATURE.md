# 📷 Live Camera Depth Mapping Feature

## 🌟 Overview

The Live Camera feature provides real-time depth map generation from your webcam feed. This cutting-edge functionality allows you to see depth estimation algorithms in action with live video input.

## ✨ Features

### **Real-time Processing**
- 📹 Live camera feed at ~30 FPS
- 🎯 Real-time depth map generation
- 🔄 Side-by-side view (Original | Depth Map)
- 📸 Instant snapshot capture

### **Smart Depth Estimation**
- 🧠 Content-aware depth generation
- 🎨 Edge detection for object boundaries
- 🌈 Brightness-based depth cues
- 🔧 Texture analysis for realistic depth

### **User-Friendly Interface**
- 🎮 Simple start/stop controls
- 📊 Real-time status monitoring
- 💾 Snapshot capture and save
- 🎨 Multiple colormap options

## 🚀 How to Use

### **1. Start Camera**
```
1. Navigate to the "Live Camera" tab
2. Click "Start Camera" button
3. Allow browser camera permissions
4. Watch real-time depth mapping!
```

### **2. Take Snapshots**
```
1. While camera is running
2. Click "Snapshot" button
3. View captured image and depth map
4. Images are displayed side-by-side
```

### **3. Stop Camera**
```
1. Click "Stop Camera" button
2. Camera feed will stop
3. Resources are properly released
```

## 🔧 Technical Details

### **Camera Configuration**
- **Resolution**: 640x480 pixels
- **Frame Rate**: 30 FPS
- **Format**: RGB color space
- **Encoding**: JPEG for web streaming

### **Depth Generation Algorithm**
```python
# Perspective-based depth (top = far, bottom = near)
base_depth = 6.0 - 4.0 * (y_coords / height)

# Edge detection for object boundaries
edges = cv2.Canny(gray_image, 50, 150)
depth_reduction = edges * 2.0  # Objects with edges are closer

# Brightness cues (darker = closer, lighter = farther)
brightness_effect = (brightness - 0.5) * 1.0

# Texture analysis for realistic variation
texture_strength = cv2.Laplacian(gray_image)
texture_depth_reduction = texture_strength * 0.5

# Final depth map
depth_map = base_depth - edge_depth - texture_depth + brightness_effect
```

### **Performance Optimizations**
- 🔄 Threaded camera capture
- 🎯 Efficient frame processing
- 📦 JPEG compression for streaming
- 🧹 Automatic resource cleanup

## 🌐 API Endpoints

### **Start Camera**
```http
POST /api/camera/start
Content-Type: application/json

{
  "camera_id": 0  // Optional, default is 0
}
```

### **Stop Camera**
```http
POST /api/camera/stop
Content-Type: application/json
```

### **Camera Status**
```http
GET /api/camera/status

Response:
{
  "is_streaming": true,
  "has_camera": true
}
```

### **Take Snapshot**
```http
POST /api/camera/snapshot

Response:
{
  "success": true,
  "original_image": "data:image/png;base64,...",
  "depth_map": "data:image/png;base64,...",
  "message": "Snapshot captured successfully"
}
```

### **Live Stream**
```http
GET /camera_feed

Returns: multipart/x-mixed-replace stream
Content: Side-by-side original and depth map images
```

## 🛠️ Browser Compatibility

### **Supported Browsers**
- ✅ Chrome 60+
- ✅ Firefox 55+
- ✅ Safari 11+
- ✅ Edge 79+

### **Required Permissions**
- 📷 Camera access
- 🌐 JavaScript enabled
- 🔒 HTTPS (for production)

## 🔍 Troubleshooting

### **Camera Not Found**
```
Issue: "Failed to start camera" error
Solutions:
1. Check camera is connected and working
2. Close other applications using camera
3. Try different camera_id (0, 1, 2...)
4. Restart browser and grant permissions
```

### **Poor Depth Quality**
```
Issue: Depth map looks unrealistic
Solutions:
1. Ensure good lighting conditions
2. Have objects at different distances
3. Avoid plain backgrounds
4. Try different camera angles
```

### **Performance Issues**
```
Issue: Slow or choppy video
Solutions:
1. Close other browser tabs
2. Reduce browser window size
3. Check system resources
4. Use Chrome for best performance
```

### **Permission Denied**
```
Issue: Browser blocks camera access
Solutions:
1. Click camera icon in address bar
2. Select "Always allow" for camera
3. Refresh the page
4. Check browser camera settings
```

## 🎯 Use Cases

### **Education**
- 📚 Computer vision demonstrations
- 🎓 Depth perception learning
- 🔬 Algorithm visualization
- 📊 Real-time analysis

### **Development**
- 🧪 Algorithm testing
- 🔧 Parameter tuning
- 📈 Performance evaluation
- 🎨 Visualization experiments

### **Research**
- 📝 Data collection
- 🔍 Behavior analysis
- 📊 Comparative studies
- 🎯 Proof of concepts

## 🔮 Future Enhancements

### **Planned Features**
- 🎛️ Adjustable depth parameters
- 📹 Video recording capability
- 🎨 Custom colormap selection
- 📊 Real-time metrics display
- 🔄 Multiple camera support
- 🎯 Advanced depth algorithms

### **Advanced Options**
- ⚙️ Manual camera settings
- 🎨 Post-processing filters
- 📐 Calibration tools
- 🎯 ROI selection
- 📊 Depth statistics

## 💡 Tips for Best Results

### **Lighting**
- 💡 Use even, diffused lighting
- 🌞 Avoid harsh shadows
- 🔆 Ensure adequate brightness
- 🌈 Avoid color temperature changes

### **Scene Setup**
- 📏 Include objects at various distances
- 🎯 Use textured surfaces
- 🔲 Avoid plain backgrounds
- 📐 Try different viewing angles

### **Camera Position**
- 📷 Keep camera stable
- 🎯 Point at interesting scenes
- 📏 Maintain reasonable distance
- 🔄 Experiment with angles

## 🎉 Success!

Your live camera depth mapping feature is now ready! Experience real-time computer vision in action and explore the fascinating world of depth estimation.

**Happy depth mapping!** 🚀📷🎯 