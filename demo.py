#!/usr/bin/env python3
"""
Demo script for the Depth Map Project.
This script demonstrates the basic functionality of the depth estimation project.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from depthmap.classical.postprocessing import DepthPostProcessor
from depthmap.utils.visualization import colorize_depth, create_side_by_side_comparison
from depthmap.eval.metrics import DepthMetrics

def create_synthetic_data():
    """Create synthetic stereo images and depth map for demonstration."""
    print("📸 Creating synthetic test data...")
    
    # Create a simple synthetic scene
    height, width = 240, 320
    
    # Create left image with some geometric patterns
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some rectangles and circles
    cv2.rectangle(left_img, (50, 50), (150, 150), (255, 100, 100), -1)
    cv2.circle(left_img, (200, 100), 40, (100, 255, 100), -1)
    cv2.rectangle(left_img, (100, 180), (250, 220), (100, 100, 255), -1)
    
    # Create a synthetic depth map
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # Add depth layers
    depth_map[50:150, 50:150] = 2.0  # Rectangle at 2m
    depth_map[60:140, 160:240] = 1.5  # Circle at 1.5m
    depth_map[180:220, 100:250] = 3.0  # Bottom rectangle at 3m
    depth_map[depth_map == 0] = 5.0  # Background at 5m
    
    # Add some noise and holes
    noise = np.random.normal(0, 0.1, depth_map.shape)
    depth_map += noise
    
    # Add some holes (invalid depth values)
    hole_mask = np.random.random(depth_map.shape) < 0.05
    depth_map[hole_mask] = 0
    
    return left_img, depth_map

def demo_postprocessing():
    """Demonstrate depth map post-processing."""
    print("\n🔧 Demonstrating Depth Map Post-processing")
    print("=" * 50)
    
    # Create synthetic data
    image, original_depth = create_synthetic_data()
    
    # Initialize post-processor
    processor = DepthPostProcessor()
    
    # Apply post-processing
    print("🔄 Applying hole filling...")
    filled_depth = processor.fill_holes(original_depth, method="inpaint")
    
    print("🔄 Applying smoothing...")
    smoothed_depth = processor.smooth_depth(filled_depth, method="bilateral")
    
    # Create visualizations
    print("🎨 Creating visualizations...")
    
    # Colorize depth maps
    original_colored = colorize_depth(original_depth, colormap="turbo")
    filled_colored = colorize_depth(filled_depth, colormap="turbo")
    smoothed_colored = colorize_depth(smoothed_depth, colormap="turbo")
    
    # Create comparison plot
    images = [image, original_colored, filled_colored, smoothed_colored]
    titles = ["Original Image", "Raw Depth", "Hole Filled", "Smoothed"]
    
    fig = create_side_by_side_comparison(
        images, titles, figsize=(16, 4), save_path="demo_postprocessing.png"
    )
    
    print("✅ Post-processing demo completed!")
    print("📁 Visualization saved as 'demo_postprocessing.png'")
    
    return original_depth, smoothed_depth

def demo_metrics():
    """Demonstrate depth evaluation metrics."""
    print("\n📊 Demonstrating Depth Evaluation Metrics")
    print("=" * 50)
    
    # Create ground truth and prediction
    gt_depth = np.random.uniform(1.0, 10.0, (100, 100))
    
    # Create a prediction with some error
    pred_depth = gt_depth + np.random.normal(0, 0.5, gt_depth.shape)
    pred_depth = np.clip(pred_depth, 0.1, 20.0)  # Ensure positive values
    
    # Initialize metrics calculator
    metrics_calc = DepthMetrics()
    
    # Compute metrics
    print("🧮 Computing evaluation metrics...")
    metrics = metrics_calc.compute_all_metrics(pred_depth, gt_depth)
    
    # Display results
    print("\n📈 Evaluation Results:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:15s}: {value:.4f}")
        else:
            print(f"{metric_name:15s}: {value}")
    
    print("✅ Metrics demo completed!")
    
    return metrics

def demo_visualization():
    """Demonstrate visualization utilities."""
    print("\n🎨 Demonstrating Visualization Utilities")
    print("=" * 50)
    
    # Create test depth map
    depth = np.random.uniform(0.5, 5.0, (150, 200))
    
    # Test different colormaps
    colormaps = ["turbo", "viridis", "plasma", "hot"]
    colored_depths = []
    
    print("🌈 Testing different colormaps...")
    for cmap in colormaps:
        colored = colorize_depth(depth, colormap=cmap)
        colored_depths.append(colored)
    
    # Create comparison
    fig = create_side_by_side_comparison(
        colored_depths, colormaps, figsize=(16, 4), save_path="demo_colormaps.png"
    )
    
    print("✅ Visualization demo completed!")
    print("📁 Colormap comparison saved as 'demo_colormaps.png'")

def main():
    """Main demo function."""
    print("🎯 Depth Map Project - Interactive Demo")
    print("=" * 60)
    print("This demo showcases the key features of the depth estimation project:")
    print("• Classical depth map post-processing")
    print("• Evaluation metrics computation")
    print("• Visualization utilities")
    print("=" * 60)
    
    try:
        # Run demonstrations
        original_depth, processed_depth = demo_postprocessing()
        metrics = demo_metrics()
        demo_visualization()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📋 Summary:")
        print("• Post-processing pipeline: ✅ Working")
        print("• Evaluation metrics: ✅ Working")
        print("• Visualization tools: ✅ Working")
        print("• Generated demo images: ✅ Available")
        
        print("\n🚀 Next Steps:")
        print("1. Check the generated visualization files:")
        print("   - demo_postprocessing.png")
        print("   - demo_colormaps.png")
        print("2. Explore Jupyter notebooks: jupyter lab notebooks/")
        print("3. Try command line tools:")
        print("   - depth-classical --help")
        print("   - depth-generative --help")
        print("4. Run evaluation: depth-eval --help")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 