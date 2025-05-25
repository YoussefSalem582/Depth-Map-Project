"""Tests for classical stereo vision pipeline."""

import numpy as np
import pytest
import cv2

from depthmap.classical import StereoDepthEstimator, StereoMethod
from depthmap.classical.calibration import StereoCalibrator


class TestStereoDepthEstimator:
    """Test cases for StereoDepthEstimator."""
    
    @pytest.fixture
    def synthetic_stereo_pair(self):
        """Create synthetic stereo image pair."""
        # Create a simple pattern
        height, width = 240, 320
        img_left = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some patterns
        cv2.rectangle(img_left, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(img_left, (200, 100), 30, (128, 128, 128), -1)
        
        # Create right image with horizontal shift (simulating disparity)
        img_right = np.roll(img_left, -10, axis=1)
        
        return img_left, img_right
    
    def test_initialization_sgbm(self):
        """Test SGBM estimator initialization."""
        estimator = StereoDepthEstimator(method="SGBM")
        assert estimator.method == StereoMethod.SGBM
        assert estimator.num_disparities == 96
        assert estimator.block_size == 11
    
    def test_initialization_bm(self):
        """Test BM estimator initialization."""
        estimator = StereoDepthEstimator(method="BM", block_size=15)
        assert estimator.method == StereoMethod.BM
        assert estimator.block_size == 15
    
    def test_invalid_parameters(self):
        """Test parameter validation."""
        # Invalid num_disparities (not divisible by 16)
        with pytest.raises(ValueError, match="num_disparities must be divisible by 16"):
            StereoDepthEstimator(num_disparities=50)
        
        # Invalid block_size (even number)
        with pytest.raises(ValueError, match="block_size must be odd"):
            StereoDepthEstimator(block_size=10)
        
        # Invalid block_size for BM (too small)
        with pytest.raises(ValueError, match="block_size must be >= 5 for BM"):
            StereoDepthEstimator(method="BM", block_size=3)
    
    def test_compute_disparity(self, synthetic_stereo_pair):
        """Test disparity computation."""
        img_left, img_right = synthetic_stereo_pair
        estimator = StereoDepthEstimator(method="BM", num_disparities=32, block_size=15)
        
        disparity = estimator.compute_disparity(img_left, img_right)
        
        assert disparity.shape == img_left.shape[:2]
        assert disparity.dtype == np.float32
        assert np.all(disparity >= 0)  # No negative disparities
    
    def test_disparity_to_depth(self):
        """Test disparity to depth conversion."""
        estimator = StereoDepthEstimator()
        
        # Create synthetic disparity map
        disparity = np.ones((100, 100)) * 10.0  # 10 pixels disparity
        focal_length = 500.0
        baseline = 0.1  # 10cm baseline
        
        depth = estimator.disparity_to_depth(disparity, focal_length, baseline)
        
        expected_depth = (focal_length * baseline) / 10.0  # 5.0 meters
        assert np.allclose(depth[depth > 0], expected_depth, rtol=1e-5)
    
    def test_estimate_depth(self, synthetic_stereo_pair):
        """Test end-to-end depth estimation."""
        img_left, img_right = synthetic_stereo_pair
        estimator = StereoDepthEstimator(method="BM", num_disparities=32, block_size=15)
        
        depth, disparity = estimator.estimate_depth(
            img_left, img_right,
            focal_length=500.0,
            baseline=0.1,
            min_depth=0.1,
            max_depth=10.0
        )
        
        assert depth.shape == img_left.shape[:2]
        assert disparity.shape == img_left.shape[:2]
        assert depth.dtype == np.float32
        assert disparity.dtype == np.float32
        
        # Check depth range
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            assert np.all(valid_depth >= 0.1)
            assert np.all(valid_depth <= 10.0)
    
    def test_parameter_update(self):
        """Test parameter updates."""
        estimator = StereoDepthEstimator(num_disparities=32)
        assert estimator.num_disparities == 32
        
        estimator.update_parameters(num_disparities=64, block_size=9)
        assert estimator.num_disparities == 64
        assert estimator.block_size == 9
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        estimator = StereoDepthEstimator(method="SGBM", num_disparities=48)
        params = estimator.get_parameters()
        
        assert params["method"] == "SGBM"
        assert params["num_disparities"] == 48
        assert "p1" in params  # SGBM-specific parameter
        assert "p2" in params


class TestStereoCalibrator:
    """Test cases for StereoCalibrator."""
    
    @pytest.fixture
    def synthetic_checkerboard_pair(self):
        """Create synthetic checkerboard images."""
        # This is a simplified version - in practice, you'd use real checkerboard images
        height, width = 480, 640
        img_left = np.zeros((height, width, 3), dtype=np.uint8)
        img_right = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a simple checkerboard pattern
        square_size = 40
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    img_left[i:i+square_size, j:j+square_size] = 255
                    img_right[i:i+square_size, j:j+square_size] = 255
        
        return img_left, img_right
    
    def test_initialization(self):
        """Test calibrator initialization."""
        calibrator = StereoCalibrator(checkerboard_size=(9, 6), square_size=0.025)
        
        assert calibrator.checkerboard_size == (9, 6)
        assert calibrator.square_size == 0.025
        assert len(calibrator.objpoints) == 0
        assert len(calibrator.imgpoints_left) == 0
        assert len(calibrator.imgpoints_right) == 0
    
    def test_object_points_generation(self):
        """Test object points generation."""
        calibrator = StereoCalibrator(checkerboard_size=(3, 2), square_size=1.0)
        
        expected_points = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 1, 0], [1, 1, 0], [2, 1, 0]
        ], dtype=np.float32)
        
        np.testing.assert_array_equal(calibrator.objp, expected_points)


@pytest.mark.parametrize("method", ["BM", "SGBM"])
def test_stereo_methods(method):
    """Test both stereo matching methods."""
    estimator = StereoDepthEstimator(method=method, num_disparities=32, block_size=15)
    assert estimator.method.value == method


def test_grayscale_input():
    """Test handling of grayscale input images."""
    estimator = StereoDepthEstimator(method="BM", num_disparities=32, block_size=15)
    
    # Create grayscale images
    img_left = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img_right = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    disparity = estimator.compute_disparity(img_left, img_right)
    assert disparity.shape == (100, 100)


def test_mismatched_image_sizes():
    """Test handling of mismatched image sizes."""
    estimator = StereoDepthEstimator(method="BM", num_disparities=32, block_size=15)
    
    img_left = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_right = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    
    # Should handle size mismatch by resizing
    disparity = estimator.compute_disparity(img_left, img_right)
    assert disparity.shape == (100, 100)


def test_zero_disparity_handling():
    """Test handling of zero disparity values."""
    estimator = StereoDepthEstimator()
    
    # Create disparity map with some zero values
    disparity = np.array([[0, 5, 10], [0, 0, 8], [12, 0, 6]], dtype=np.float32)
    
    depth = estimator.disparity_to_depth(disparity, 500.0, 0.1)
    
    # Zero disparity should result in zero depth
    assert depth[0, 0] == 0
    assert depth[1, 0] == 0
    assert depth[1, 1] == 0
    
    # Non-zero disparity should result in valid depth
    assert depth[0, 1] > 0
    assert depth[0, 2] > 0 