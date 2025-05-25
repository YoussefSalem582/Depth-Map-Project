#!/usr/bin/env python3
"""
Simple test script to run the depth map project.
"""

import sys
import os
import numpy as np
import cv2

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_classical_import():
    """Test importing classical depth estimation modules."""
    try:
        from depthmap.classical import postprocessing
        print("✓ Successfully imported classical postprocessing module")
        
        # Test the post-processor
        processor = postprocessing.DepthPostProcessor()
        print("✓ Successfully created DepthPostProcessor instance")
        
        # Create a simple test depth map
        test_depth = np.random.rand(100, 100) * 10.0
        test_depth[20:30, 20:30] = 0  # Add some holes
        
        # Test hole filling
        filled_depth = processor.fill_holes(test_depth)
        print(f"✓ Hole filling test passed. Original shape: {test_depth.shape}, Filled shape: {filled_depth.shape}")
        
        # Test smoothing
        smoothed_depth = processor.smooth_depth(test_depth)
        print(f"✓ Smoothing test passed. Original shape: {test_depth.shape}, Smoothed shape: {smoothed_depth.shape}")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import classical modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing classical modules: {e}")
        return False

def test_utils_import():
    """Test importing utility modules."""
    try:
        from depthmap.utils import visualization, io, config
        print("✓ Successfully imported utils modules (visualization, io, config)")
        return True
    except ImportError as e:
        print(f"✗ Failed to import utils modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing utils modules: {e}")
        return False

def test_eval_import():
    """Test importing evaluation modules."""
    try:
        from depthmap.eval import metrics, plotting
        print("✓ Successfully imported eval modules (metrics, plotting)")
        return True
    except ImportError as e:
        print(f"✗ Failed to import eval modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing eval modules: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 Depth Map Project - Test Run")
    print("=" * 50)
    
    # Test OpenCV
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_classical_import():
        tests_passed += 1
    
    if test_utils_import():
        tests_passed += 1
    
    if test_eval_import():
        tests_passed += 1
    
    print()
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The project is ready to run.")
        print()
        print("Next steps:")
        print("1. Run 'jupyter lab' to explore the notebooks")
        print("2. Try the command line tools: depth-classical, depth-generative")
        print("3. Run 'streamlit run app/depth_estimation_app.py' for the web app")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 