#!/usr/bin/env python3
"""
Flask Web Application for Depth Map Project
A simple web interface for depth estimation and visualization.
"""

import os
import sys
import io
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
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

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

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
    print("🎯 Depth Map Project - Flask Web Application")
    print("=" * 50)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='127.0.0.1', port=5000) 