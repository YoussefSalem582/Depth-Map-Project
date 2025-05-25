"""Stereo depth estimation using classical computer vision methods."""

import logging
from enum import Enum
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StereoMethod(Enum):
    """Supported stereo matching methods."""
    BM = "BM"
    SGBM = "SGBM"


class StereoDepthEstimator:
    """Classical stereo depth estimation using OpenCV algorithms."""
    
    def __init__(
        self,
        method: Union[str, StereoMethod] = StereoMethod.SGBM,
        num_disparities: int = 96,
        block_size: int = 11,
        min_disparity: int = 0,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 32,
        disp12_max_diff: int = 1,
        pre_filter_cap: int = 63,
        pre_filter_size: int = 5,
        texture_threshold: int = 10,
        smaller_block_size: int = 0,
        p1: Optional[int] = None,
        p2: Optional[int] = None,
        mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    ):
        """Initialize stereo depth estimator.
        
        Args:
            method: Stereo matching method ('BM' or 'SGBM').
            num_disparities: Maximum disparity minus minimum disparity. Must be divisible by 16.
            block_size: Matched block size. Must be odd and >= 1.
            min_disparity: Minimum possible disparity value.
            uniqueness_ratio: Margin in percentage by which the best computed cost function value 
                            should "win" the second best value to consider the found match correct.
            speckle_window_size: Maximum size of smooth disparity regions to consider their noise 
                               speckles and invalidate.
            speckle_range: Maximum disparity variation within each connected component.
            disp12_max_diff: Maximum allowed difference in the left-right disparity check.
            pre_filter_cap: Truncation value for the prefiltered image pixels.
            pre_filter_size: Averaging window size for prefiltering.
            texture_threshold: Minimum texture threshold for BM algorithm.
            smaller_block_size: Smaller block size for SGBM algorithm.
            p1: First parameter controlling the disparity smoothness (SGBM only).
            p2: Second parameter controlling the disparity smoothness (SGBM only).
            mode: SGBM mode (SGBM only).
        """
        self.method = StereoMethod(method) if isinstance(method, str) else method
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_disparity = min_disparity
        self.uniqueness_ratio = uniqueness_ratio
        self.speckle_window_size = speckle_window_size
        self.speckle_range = speckle_range
        self.disp12_max_diff = disp12_max_diff
        self.pre_filter_cap = pre_filter_cap
        self.pre_filter_size = pre_filter_size
        self.texture_threshold = texture_threshold
        self.smaller_block_size = smaller_block_size
        self.mode = mode
        
        # Auto-calculate P1 and P2 for SGBM if not provided
        if self.method == StereoMethod.SGBM:
            if p1 is None:
                self.p1 = 8 * 3 * block_size * block_size
            else:
                self.p1 = p1
                
            if p2 is None:
                self.p2 = 32 * 3 * block_size * block_size
            else:
                self.p2 = p2
        
        # Validate parameters
        self._validate_parameters()
        
        # Create stereo matcher
        self.stereo_matcher = self._create_stereo_matcher()
        
        logger.info(f"Initialized {self.method.value} stereo depth estimator")
    
    def _validate_parameters(self) -> None:
        """Validate stereo matching parameters."""
        if self.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16")
        
        if self.block_size % 2 == 0 or self.block_size < 1:
            raise ValueError("block_size must be odd and >= 1")
        
        if self.method == StereoMethod.BM and self.block_size < 5:
            raise ValueError("block_size must be >= 5 for BM algorithm")
    
    def _create_stereo_matcher(self) -> Union[cv2.StereoBM, cv2.StereoSGBM]:
        """Create the appropriate stereo matcher based on the method."""
        if self.method == StereoMethod.BM:
            stereo = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=self.block_size
            )
            stereo.setMinDisparity(self.min_disparity)
            stereo.setUniquenessRatio(self.uniqueness_ratio)
            stereo.setSpeckleWindowSize(self.speckle_window_size)
            stereo.setSpeckleRange(self.speckle_range)
            stereo.setDisp12MaxDiff(self.disp12_max_diff)
            stereo.setPreFilterCap(self.pre_filter_cap)
            stereo.setPreFilterSize(self.pre_filter_size)
            stereo.setTextureThreshold(self.texture_threshold)
            stereo.setSmallerBlockSize(self.smaller_block_size)
            
        elif self.method == StereoMethod.SGBM:
            stereo = cv2.StereoSGBM_create(
                minDisparity=self.min_disparity,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=self.p1,
                P2=self.p2,
                disp12MaxDiff=self.disp12_max_diff,
                uniquenessRatio=self.uniqueness_ratio,
                speckleWindowSize=self.speckle_window_size,
                speckleRange=self.speckle_range,
                preFilterCap=self.pre_filter_cap,
                mode=self.mode
            )
        
        return stereo
    
    def compute_disparity(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Compute disparity map from stereo image pair.
        
        Args:
            img_left: Left camera image.
            img_right: Right camera image.
            normalize: Whether to normalize disparity values.
            
        Returns:
            Disparity map as numpy array.
        """
        # Convert to grayscale if needed
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left.copy()
            
        if len(img_right.shape) == 3:
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = img_right.copy()
        
        # Ensure images have the same size
        if gray_left.shape != gray_right.shape:
            logger.warning("Left and right images have different sizes. Resizing right image.")
            gray_right = cv2.resize(gray_right, (gray_left.shape[1], gray_left.shape[0]))
        
        # Compute disparity
        logger.debug(f"Computing disparity using {self.method.value}")
        disparity = self.stereo_matcher.compute(gray_left, gray_right)
        
        # Convert to float32 and normalize
        disparity = disparity.astype(np.float32)
        
        if normalize:
            # OpenCV returns disparity in fixed-point format (multiply by 16)
            disparity = disparity / 16.0
        
        # Set invalid disparities to 0
        disparity[disparity < 0] = 0
        
        return disparity
    
    def disparity_to_depth(
        self,
        disparity: np.ndarray,
        focal_length: float,
        baseline: float,
        min_depth: float = 0.1,
        max_depth: float = 100.0
    ) -> np.ndarray:
        """Convert disparity map to depth map.
        
        Args:
            disparity: Disparity map.
            focal_length: Camera focal length in pixels.
            baseline: Stereo baseline in same units as desired depth.
            min_depth: Minimum valid depth value.
            max_depth: Maximum valid depth value.
            
        Returns:
            Depth map in same units as baseline.
        """
        # Avoid division by zero
        valid_mask = disparity > 0
        depth = np.zeros_like(disparity)
        
        # Depth = (focal_length * baseline) / disparity
        depth[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
        
        # Clamp depth values
        depth = np.clip(depth, min_depth, max_depth)
        
        # Set invalid depths to 0
        depth[~valid_mask] = 0
        
        return depth
    
    def estimate_depth(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        focal_length: float,
        baseline: float,
        min_depth: float = 0.1,
        max_depth: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate depth from stereo image pair.
        
        Args:
            img_left: Left camera image.
            img_right: Right camera image.
            focal_length: Camera focal length in pixels.
            baseline: Stereo baseline in same units as desired depth.
            min_depth: Minimum valid depth value.
            max_depth: Maximum valid depth value.
            
        Returns:
            Tuple of (depth_map, disparity_map).
        """
        # Compute disparity
        disparity = self.compute_disparity(img_left, img_right)
        
        # Convert to depth
        depth = self.disparity_to_depth(disparity, focal_length, baseline, min_depth, max_depth)
        
        logger.info(f"Estimated depth map with {np.sum(depth > 0)} valid pixels")
        
        return depth, disparity
    
    def get_parameters(self) -> dict:
        """Get current stereo matching parameters.
        
        Returns:
            Dictionary of current parameters.
        """
        params = {
            "method": self.method.value,
            "num_disparities": self.num_disparities,
            "block_size": self.block_size,
            "min_disparity": self.min_disparity,
            "uniqueness_ratio": self.uniqueness_ratio,
            "speckle_window_size": self.speckle_window_size,
            "speckle_range": self.speckle_range,
            "disp12_max_diff": self.disp12_max_diff,
            "pre_filter_cap": self.pre_filter_cap,
            "pre_filter_size": self.pre_filter_size,
            "texture_threshold": self.texture_threshold,
            "smaller_block_size": self.smaller_block_size,
        }
        
        if self.method == StereoMethod.SGBM:
            params.update({
                "p1": self.p1,
                "p2": self.p2,
                "mode": self.mode,
            })
        
        return params
    
    def update_parameters(self, **kwargs) -> None:
        """Update stereo matching parameters and recreate matcher.
        
        Args:
            **kwargs: Parameter updates.
        """
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown parameter: {key}")
        
        # Validate and recreate matcher
        self._validate_parameters()
        self.stereo_matcher = self._create_stereo_matcher()
        
        logger.info("Updated stereo matcher parameters") 