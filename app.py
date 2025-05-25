#!/usr/bin/env python3
"""
Flask Web Application for Depth Map Project
A simple web interface for depth estimation and visualization with live camera support.
"""

import os
import sys
import io
import base64
import threading
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from PIL import Image

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from depthmap.classical.postprocessing import DepthPostProcessor
from depthmap.utils.visualization import colorize_depth
from depthmap.eval.metrics import DepthMetrics

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global processor instance
processor = DepthPostProcessor()

# Camera configuration
class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_streaming = False
        self.frame = None
        self.depth_frame = None
        self.lock = threading.Lock()
        
    def start_camera(self, camera_id=0):
        """Start camera capture."""
        try:
            # Stop any existing camera first
            if self.camera:
                self.camera.release()
                
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                print(f"Failed to open camera {camera_id}")
                return False
            
            # Test if we can read a frame
            ret, test_frame = self.camera.read()
            if not ret:
                print(f"Failed to read from camera {camera_id}")
                self.camera.release()
                self.camera = None
                return False
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_streaming = True
            print(f"Camera {camera_id} started successfully")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            if self.camera:
                self.camera.release()
                self.camera = None
            return False
    
    def stop_camera(self):
        """Stop camera capture."""
        self.is_streaming = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def get_frame(self):
        """Get current frame from camera."""
        if not self.camera or not self.is_streaming:
            return None, None
        
        ret, frame = self.camera.read()
        if not ret:
            return None, None
        
        # Generate depth map for the frame
        depth_map = create_image_based_depth(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with self.lock:
            self.frame = frame.copy()
            self.depth_frame = depth_map.copy()
        
        return frame, depth_map

# Global camera manager
camera_manager = CameraManager()

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_synthetic_depth(width=320, height=240):
    """Create a realistic synthetic depth map for demonstration."""
    # Create coordinate meshgrids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create base depth map with perspective (top = far, bottom = near)
    depth_map = 8.0 - 3.0 * (y_coords / height)  # 8m at top, 5m at bottom
    
    # Add circular objects at different depths
    # Object 1: Close circular object (person/object in foreground)
    circle1_x, circle1_y = width * 0.3, height * 0.6
    circle1_radius = min(width, height) * 0.15
    dist1 = np.sqrt((x_coords - circle1_x)**2 + (y_coords - circle1_y)**2)
    circle1_mask = dist1 < circle1_radius
    depth_map[circle1_mask] = 1.5 + 0.3 * (dist1[circle1_mask] / circle1_radius)
    
    # Object 2: Medium distance object
    circle2_x, circle2_y = width * 0.7, height * 0.4
    circle2_radius = min(width, height) * 0.12
    dist2 = np.sqrt((x_coords - circle2_x)**2 + (y_coords - circle2_y)**2)
    circle2_mask = dist2 < circle2_radius
    depth_map[circle2_mask] = 3.0 + 0.5 * (dist2[circle2_mask] / circle2_radius)
    
    # Add ground plane effect (perspective)
    ground_mask = y_coords > height * 0.7
    ground_depth = 2.0 + 4.0 * ((y_coords - height * 0.7) / (height * 0.3))
    depth_map[ground_mask] = np.minimum(depth_map[ground_mask], ground_depth[ground_mask])
    
    # Add smooth noise for realism
    noise = np.random.normal(0, 0.05, depth_map.shape)
    depth_map += noise
    
    # Apply Gaussian smoothing for more realistic transitions
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 1.0)
    
    # Add some small holes (occlusions)
    hole_mask = np.random.random(depth_map.shape) < 0.02
    depth_map[hole_mask] = 0
    
    # Ensure positive values
    depth_map = np.clip(depth_map, 0.1, 10.0)
    
    return depth_map.astype(np.float32)

def create_image_based_depth(image_rgb):
    """Create a depth map based on image content using simple heuristics."""
    h, w = image_rgb.shape[:2]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Detect edges (objects tend to be closer)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Create base depth map with perspective (top = far, bottom = near)
    y_coords = np.arange(h).reshape(-1, 1)
    base_depth = 6.0 - 4.0 * (y_coords / h)  # 6m at top, 2m at bottom
    base_depth = np.broadcast_to(base_depth, (h, w))
    
    # Objects with edges are typically closer
    edge_depth_reduction = edges.astype(np.float32) / 255.0 * 2.0
    depth_map = base_depth - edge_depth_reduction
    
    # Use brightness as depth cue (darker = closer, lighter = farther)
    brightness = gray.astype(np.float32) / 255.0
    brightness_effect = (brightness - 0.5) * 1.0  # ±0.5m based on brightness
    depth_map += brightness_effect
    
    # Add some texture-based depth variation
    # High frequency areas (textures) tend to be closer
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_strength = np.abs(laplacian) / np.max(np.abs(laplacian))
    texture_depth_reduction = texture_strength * 0.5
    depth_map -= texture_depth_reduction
    
    # Smooth the depth map
    depth_map = cv2.GaussianBlur(depth_map, (7, 7), 1.5)
    
    # Add realistic noise
    noise = np.random.normal(0, 0.1, depth_map.shape)
    depth_map += noise
    
    # Ensure reasonable depth range
    depth_map = np.clip(depth_map, 0.5, 8.0)
    
    return depth_map.astype(np.float32)

def numpy_to_base64(array, format='PNG'):
    """Convert numpy array to base64 string for web display."""
    if len(array.shape) == 2:
        # Grayscale image
        img = Image.fromarray((array * 255).astype(np.uint8), mode='L')
    else:
        # RGB image
        img = Image.fromarray(array.astype(np.uint8), mode='RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_base64}"

def generate_camera_frames():
    """Generate camera frames for streaming."""
    print("Starting camera frame generation...")
    frame_count = 0
    
    while camera_manager.is_streaming:
        frame, depth_map = camera_manager.get_frame()
        
        if frame is not None and depth_map is not None:
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"Generated {frame_count} frames")
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create depth visualization
            depth_colored = colorize_depth(depth_map, colormap="turbo")
            
            # Create side-by-side view
            combined = np.hstack([frame_rgb, depth_colored])
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            print("No frame available, waiting...")
        
        time.sleep(0.033)  # ~30 FPS
    
    print("Camera frame generation stopped.")

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/debug')
def debug():
    """Camera debug page."""
    return render_template('camera_debug.html')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera streaming."""
    try:
        # Handle both JSON and non-JSON requests
        camera_id = 0
        if request.is_json and request.json:
            camera_id = request.json.get('camera_id', 0)
        
        if camera_manager.start_camera(camera_id):
            return jsonify({'success': True, 'message': 'Camera started successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to start camera'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera streaming."""
    try:
        camera_manager.stop_camera()
        return jsonify({'success': True, 'message': 'Camera stopped successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get camera status."""
    return jsonify({
        'is_streaming': camera_manager.is_streaming,
        'has_camera': camera_manager.camera is not None
    })

@app.route('/camera_feed')
def camera_feed():
    """Video streaming route."""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/snapshot', methods=['POST'])
def camera_snapshot():
    """Take a snapshot from the camera."""
    try:
        if not camera_manager.is_streaming:
            return jsonify({'success': False, 'error': 'Camera is not running'})
        
        with camera_manager.lock:
            if camera_manager.frame is not None and camera_manager.depth_frame is not None:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(camera_manager.frame, cv2.COLOR_BGR2RGB)
                
                # Create depth visualization
                depth_colored = colorize_depth(camera_manager.depth_frame, colormap="turbo")
                
                # Convert to base64
                results = {
                    'original_image': numpy_to_base64(frame_rgb),
                    'depth_map': numpy_to_base64(depth_colored),
                    'success': True,
                    'message': 'Snapshot captured successfully'
                }
                
                return jsonify(results)
            else:
                return jsonify({'success': False, 'error': 'No frame available'})
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/demo', methods=['POST'])
def demo():
    """Generate and process a demo depth map."""
    try:
        # Create synthetic data
        original_depth = create_synthetic_depth()
        
        # Apply post-processing
        filled_depth = processor.fill_holes(original_depth, method="inpaint")
        smoothed_depth = processor.smooth_depth(filled_depth, method="bilateral")
        
        # Create visualizations
        original_colored = colorize_depth(original_depth, colormap="turbo")
        filled_colored = colorize_depth(filled_depth, colormap="turbo")
        smoothed_colored = colorize_depth(smoothed_depth, colormap="turbo")
        
        # Convert to base64 for web display
        results = {
            'original': numpy_to_base64(original_colored),
            'filled': numpy_to_base64(filled_colored),
            'smoothed': numpy_to_base64(smoothed_colored),
            'success': True
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process image."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'})
        
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Load and process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not read image'})
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a more realistic depth map based on image content
        h, w = image.shape[:2]
        synthetic_depth = create_image_based_depth(image_rgb)
        
        # Apply post-processing
        processed_depth = processor.process_depth_map(synthetic_depth)
        
        # Create visualizations
        depth_colored = colorize_depth(processed_depth, colormap="turbo")
        
        # Convert to base64
        results = {
            'original_image': numpy_to_base64(image_rgb),
            'depth_map': numpy_to_base64(depth_colored),
            'success': True,
            'message': 'Image processed successfully (using synthetic depth for demo)'
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/evaluate', methods=['POST'])
def evaluate_depth():
    """Evaluate depth estimation metrics."""
    try:
        # Create sample data for evaluation
        gt_depth = np.random.uniform(1.0, 10.0, (100, 100))
        pred_depth = gt_depth + np.random.normal(0, 0.5, gt_depth.shape)
        pred_depth = np.clip(pred_depth, 0.1, 20.0)
        
        # Calculate metrics
        metrics_calc = DepthMetrics()
        metrics = metrics_calc.compute_all_metrics(pred_depth, gt_depth)
        
        # Format metrics for display
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics[key] = round(value, 4)
            else:
                formatted_metrics[key] = value
        
        return jsonify({
            'success': True,
            'metrics': formatted_metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/colormaps', methods=['GET'])
def get_colormaps():
    """Get available colormaps demonstration."""
    try:
        # Create a more interesting test depth map
        depth = create_synthetic_depth(200, 150)
        
        # Test different colormaps
        colormaps = ["turbo", "viridis", "plasma", "hot", "cool", "spring"]
        results = {}
        
        for cmap in colormaps:
            colored = colorize_depth(depth, colormap=cmap)
            results[cmap] = numpy_to_base64(colored)
        
        return jsonify({
            'success': True,
            'colormaps': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🎯 Depth Map Project - Flask Web Application with Live Camera")
    print("=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("Features:")
    print("  📷 Live Camera Feed with Real-time Depth Maps")
    print("  📱 Interactive Demo")
    print("  📤 Upload & Process Images")
    print("  📊 Evaluation Metrics")
    print("  🎨 Colormap Gallery")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
    finally:
        # Cleanup camera on exit
        camera_manager.stop_camera() 