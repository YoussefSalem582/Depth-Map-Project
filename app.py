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

# Try to import MiDaS for advanced depth estimation
try:
    from depthmap.generative.midas import MiDaSDepthEstimator
    from depthmap.utils.model_manager import get_model_manager
    MIDAS_AVAILABLE = True
    print("MiDaS model available for enhanced depth estimation")
except ImportError as e:
    MIDAS_AVAILABLE = False
    print(f"MiDaS not available: {e}. Using classical methods only.")

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global processor instance
processor = DepthPostProcessor()

# Global MiDaS instance (if available)
midas_estimator = None
model_manager = None

if MIDAS_AVAILABLE:
    try:
        # Initialize model manager
        model_manager = get_model_manager()
        print(f"Model manager initialized. Cache dir: {model_manager.cache_dir}")
        
        # Check if model is already cached
        if model_manager.is_model_cached("dpt_large"):
            print("Found cached MiDaS model, loading instantly...")
        else:
            print("MiDaS model not cached, will download on first use...")
        
        # Initialize MiDaS with caching enabled
        midas_estimator = MiDaSDepthEstimator(model_name="DPT_Large", use_cache=True)
        print("MiDaS DPT_Large model loaded successfully with caching")
    except Exception as e:
        print(f"Failed to load MiDaS model: {e}")
        MIDAS_AVAILABLE = False

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
        depth_map = create_enhanced_depth(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
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

def create_enhanced_synthetic_depth(width=320, height=240):
    """Create a highly realistic synthetic depth map with advanced patterns."""
    # Create coordinate meshgrids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create base depth map with realistic perspective
    depth_map = 10.0 - 6.0 * (y_coords / height)  # 10m at top, 4m at bottom
    
    # Add multiple realistic objects with varying shapes and depths
    
    # Object 1: Large foreground object (person/furniture)
    obj1_x, obj1_y = width * 0.25, height * 0.65
    obj1_radius = min(width, height) * 0.18
    dist1 = np.sqrt((x_coords - obj1_x)**2 + (y_coords - obj1_y)**2)
    obj1_mask = dist1 < obj1_radius
    # Create 3D-like depth variation within object
    depth_variation = 0.4 * np.cos(dist1[obj1_mask] / obj1_radius * np.pi)
    depth_map[obj1_mask] = 1.2 + depth_variation
    
    # Object 2: Medium distance object with irregular shape
    obj2_x, obj2_y = width * 0.7, height * 0.45
    obj2_radius = min(width, height) * 0.12
    dist2 = np.sqrt((x_coords - obj2_x)**2 + (y_coords - obj2_y)**2)
    # Create irregular shape using multiple circles
    obj2_mask = (dist2 < obj2_radius) | \
                (np.sqrt((x_coords - obj2_x - 15)**2 + (y_coords - obj2_y + 10)**2) < obj2_radius * 0.7)
    depth_map[obj2_mask] = 2.8 + 0.3 * np.sin(dist2[obj2_mask] / obj2_radius * 2 * np.pi)
    
    # Object 3: Background object
    obj3_x, obj3_y = width * 0.5, height * 0.25
    obj3_radius = min(width, height) * 0.15
    dist3 = np.sqrt((x_coords - obj3_x)**2 + (y_coords - obj3_y)**2)
    obj3_mask = dist3 < obj3_radius
    depth_map[obj3_mask] = 6.0 + 0.5 * (dist3[obj3_mask] / obj3_radius)
    
    # Add realistic ground plane with perspective distortion
    ground_mask = y_coords > height * 0.75
    ground_depth = 1.5 + 3.0 * ((y_coords - height * 0.75) / (height * 0.25))**1.5
    depth_map[ground_mask] = np.minimum(depth_map[ground_mask], ground_depth[ground_mask])
    
    # Add wall/ceiling effects
    wall_left = x_coords < width * 0.1
    wall_right = x_coords > width * 0.9
    ceiling = y_coords < height * 0.1
    
    depth_map[wall_left] = np.maximum(depth_map[wall_left], 3.0 + x_coords[wall_left] / width * 2.0)
    depth_map[wall_right] = np.maximum(depth_map[wall_right], 3.0 + (width - x_coords[wall_right]) / width * 2.0)
    depth_map[ceiling] = np.maximum(depth_map[ceiling], 8.0)
    
    # Add realistic surface textures and variations
    # High-frequency noise for surface texture
    texture_noise = np.random.normal(0, 0.02, depth_map.shape)
    depth_map += texture_noise
    
    # Low-frequency variations for surface undulations
    low_freq_x = np.sin(x_coords / width * 4 * np.pi) * 0.1
    low_freq_y = np.cos(y_coords / height * 3 * np.pi) * 0.08
    depth_map += low_freq_x + low_freq_y
    
    # Apply advanced smoothing for realistic transitions
    depth_map = cv2.GaussianBlur(depth_map, (7, 7), 1.5)
    
    # Add some occlusions and depth discontinuities
    occlusion_mask = np.random.random(depth_map.shape) < 0.015
    depth_map[occlusion_mask] = 0
    
    # Ensure realistic depth range
    depth_map = np.clip(depth_map, 0.1, 12.0)
    
    return depth_map.astype(np.float32)

def create_enhanced_depth(image_rgb):
    """Create enhanced depth map using MiDaS if available, otherwise advanced classical methods."""
    if MIDAS_AVAILABLE and midas_estimator is not None:
        try:
            # Use MiDaS for state-of-the-art depth estimation
            depth_map = midas_estimator.predict(image_rgb)
            
            # Convert MiDaS relative depth to metric depth (approximate)
            # MiDaS outputs relative depth, we convert to reasonable metric values
            depth_min, depth_max = depth_map.min(), depth_map.max()
            if depth_max > depth_min:
                # Map to 0.5m - 10m range
                depth_map = 0.5 + (depth_map - depth_min) / (depth_max - depth_min) * 9.5
            
            # Apply post-processing for even better results
            depth_map = processor.process_depth_map(depth_map, image_rgb, 
                                                  fill_holes=True, smooth=True, 
                                                  enhance_edges=True, multi_scale=True)
            
            return depth_map
            
        except Exception as e:
            print(f"MiDaS prediction failed: {e}, falling back to classical method")
    
    # Enhanced classical depth estimation
    return create_advanced_image_based_depth(image_rgb)

def create_advanced_image_based_depth(image_rgb):
    """Create advanced depth map based on image content using multiple cues."""
    h, w = image_rgb.shape[:2]
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    # 1. Edge-based depth cues (objects with edges are typically closer)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edge_strength = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 5.0)
    
    # 2. Create base perspective depth map
    y_coords = np.arange(h).reshape(-1, 1)
    base_depth = 8.0 - 5.0 * (y_coords / h)**1.2  # Non-linear perspective
    base_depth = np.broadcast_to(base_depth, (h, w))
    
    # 3. Brightness and contrast cues
    brightness = gray.astype(np.float32) / 255.0
    # Darker areas tend to be closer (shadows, recesses)
    brightness_depth = (0.7 - brightness) * 2.0
    
    # 4. Color saturation cues (more saturated = closer)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    saturation_depth = (saturation - 0.5) * -1.5
    
    # 5. Texture and detail analysis
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_strength = np.abs(laplacian)
    texture_strength = cv2.GaussianBlur(texture_strength, (5, 5), 1.0)
    texture_strength = texture_strength / np.max(texture_strength)
    texture_depth = texture_strength * -1.0  # High texture = closer
    
    # 6. Superpixel-based depth estimation
    try:
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        
        # Create superpixels
        segments = slic(image_rgb, n_segments=100, compactness=10, sigma=1)
        
        # Analyze each superpixel
        segment_depths = np.zeros_like(segments, dtype=np.float32)
        for region in regionprops(segments + 1):  # +1 because regionprops expects 1-based labels
            mask = segments == (region.label - 1)
            
            # Combine multiple cues for this region
            region_y = region.centroid[0] / h  # Vertical position
            region_brightness = np.mean(brightness[mask])
            region_saturation = np.mean(saturation[mask])
            region_texture = np.mean(texture_strength[mask])
            
            # Weighted combination
            region_depth = (8.0 - 5.0 * region_y**1.2 +  # Perspective
                          (0.7 - region_brightness) * 1.5 +  # Brightness
                          (region_saturation - 0.5) * -1.0 +  # Saturation
                          region_texture * -0.8)  # Texture
            
            segment_depths[mask] = region_depth
            
        depth_map = segment_depths
        
    except ImportError:
        # Fallback if scikit-image not available
        depth_map = (base_depth + 
                    brightness_depth * 0.3 + 
                    saturation_depth * 0.2 + 
                    texture_depth * 0.2)
    
    # 7. Edge-guided refinement
    edge_mask = edge_strength / 255.0
    edge_depth_reduction = edge_mask * 1.5
    depth_map -= edge_depth_reduction
    
    # 8. Apply advanced post-processing
    depth_map = processor.process_depth_map(depth_map, image_rgb,
                                          fill_holes=True, smooth=True,
                                          enhance_edges=True, multi_scale=False)
    
    # 9. Final depth range normalization
    depth_map = np.clip(depth_map, 0.3, 10.0)
    
    return depth_map.astype(np.float32)

def create_synthetic_depth(width=320, height=240):
    """Create a realistic synthetic depth map for demonstration."""
    return create_enhanced_synthetic_depth(width, height)

def create_image_based_depth(image_rgb):
    """Create a depth map based on image content using enhanced methods."""
    return create_enhanced_depth(image_rgb)

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
            
            # Create enhanced depth visualization
            depth_colored = colorize_depth(depth_map, colormap="turbo",
                                         enhance_contrast=True, apply_gamma=True)
            
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
                
                # Create enhanced depth visualization
                depth_colored = colorize_depth(camera_manager.depth_frame, colormap="turbo",
                                             enhance_contrast=True, apply_gamma=True)
                
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
        # Create enhanced synthetic data
        original_depth = create_enhanced_synthetic_depth()
        
        # Apply advanced post-processing pipeline
        filled_depth = processor.fill_holes(original_depth, method="morphological")
        smoothed_depth = processor.smooth_depth(filled_depth, method="edge_preserving")
        final_depth = processor.multi_scale_refinement(smoothed_depth)
        
        # Create enhanced visualizations
        original_colored = colorize_depth(original_depth, colormap="turbo", 
                                        enhance_contrast=True, apply_gamma=True)
        filled_colored = colorize_depth(filled_depth, colormap="turbo",
                                      enhance_contrast=True, apply_gamma=True)
        final_colored = colorize_depth(final_depth, colormap="turbo",
                                     enhance_contrast=True, apply_gamma=True)
        
        # Convert to base64 for web display
        results = {
            'original': numpy_to_base64(original_colored),
            'filled': numpy_to_base64(filled_colored),
            'smoothed': numpy_to_base64(final_colored),
            'success': True,
            'message': 'Enhanced depth map generated with advanced processing'
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
        
        # Create enhanced depth map using advanced methods
        h, w = image.shape[:2]
        enhanced_depth = create_enhanced_depth(image_rgb)
        
        # Apply advanced post-processing
        processed_depth = processor.process_depth_map(enhanced_depth, image_rgb,
                                                    fill_holes=True, smooth=True,
                                                    enhance_edges=True, multi_scale=True)
        
        # Create enhanced visualizations
        depth_colored = colorize_depth(processed_depth, colormap="turbo",
                                     enhance_contrast=True, apply_gamma=True)
        
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

@app.route('/api/models/cache/info', methods=['GET'])
def get_cache_info():
    """Get model cache information."""
    try:
        if model_manager is None:
            return jsonify({'success': False, 'error': 'Model manager not available'})
        
        cache_info = model_manager.get_cache_info()
        cache_stats = model_manager.get_cache_stats()
        
        return jsonify({
            'success': True,
            'cache_info': cache_info,
            'cache_stats': cache_stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/cache/clear', methods=['POST'])
def clear_cache():
    """Clear model cache."""
    try:
        if model_manager is None:
            return jsonify({'success': False, 'error': 'Model manager not available'})
        
        # Get confirmation from request
        confirm = request.json.get('confirm', False) if request.is_json else False
        
        if not confirm:
            return jsonify({
                'success': False, 
                'error': 'Confirmation required. Send {"confirm": true} to clear cache.'
            })
        
        success = model_manager.clear_cache(confirm=True)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Cache cleared successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to clear cache'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/cache/optimize', methods=['POST'])
def optimize_cache():
    """Optimize model cache by removing corrupted files."""
    try:
        if model_manager is None:
            return jsonify({'success': False, 'error': 'Model manager not available'})
        
        results = model_manager.optimize_cache()
        
        return jsonify({
            'success': True,
            'optimization_results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/export/<model_name>', methods=['POST'])
def export_model(model_name):
    """Export a cached model."""
    try:
        if model_manager is None:
            return jsonify({'success': False, 'error': 'Model manager not available'})
        
        data = request.json if request.is_json else {}
        export_path = data.get('export_path')
        format_type = data.get('format', 'pt')
        
        if not export_path:
            return jsonify({'success': False, 'error': 'export_path required'})
        
        success = model_manager.export_model(model_name, export_path, format_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Model {model_name} exported to {export_path}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to export model {model_name}'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/status', methods=['GET'])
def get_model_status():
    """Get status of loaded models."""
    try:
        status = {
            'midas_available': MIDAS_AVAILABLE,
            'midas_loaded': midas_estimator is not None,
            'model_manager_available': model_manager is not None,
            'cached_models': []
        }
        
        if model_manager:
            status['cached_models'] = model_manager.list_cached_models()
            status['cache_stats'] = model_manager.get_cache_stats()
        
        if midas_estimator:
            status['midas_info'] = midas_estimator.get_model_info()
            if hasattr(midas_estimator, 'get_cache_info'):
                status['midas_cache_info'] = midas_estimator.get_cache_info()
        
        return jsonify({
            'success': True,
            'status': status
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