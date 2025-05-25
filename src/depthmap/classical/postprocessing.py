"""Depth map post-processing utilities."""

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class DepthPostProcessor:
    """Post-processing utilities for depth maps."""
    
    def __init__(self):
        """Initialize depth post-processor."""
        logger.info("Depth post-processor initialized")
    
    def fill_holes(
        self,
        depth: np.ndarray,
        method: str = "inpaint",
        kernel_size: int = 5,
        max_hole_size: int = 100
    ) -> np.ndarray:
        """Fill holes in depth map.
        
        Args:
            depth: Input depth map.
            method: Hole filling method ('inpaint', 'interpolate', 'median').
            kernel_size: Kernel size for morphological operations.
            max_hole_size: Maximum hole size to fill (in pixels).
            
        Returns:
            Depth map with filled holes.
        """
        # Create mask of invalid pixels
        invalid_mask = (depth == 0) | np.isnan(depth) | np.isinf(depth)
        
        if not np.any(invalid_mask):
            return depth.copy()
        
        filled_depth = depth.copy()
        
        if method == "inpaint":
            # Use OpenCV inpainting
            depth_uint8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask_uint8 = invalid_mask.astype(np.uint8) * 255
            
            inpainted = cv2.inpaint(depth_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
            
            # Convert back to original scale
            depth_min, depth_max = depth[~invalid_mask].min(), depth[~invalid_mask].max()
            filled_depth[invalid_mask] = (inpainted[invalid_mask] / 255.0) * (depth_max - depth_min) + depth_min
            
        elif method == "interpolate":
            # Simple interpolation using valid neighbors
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Iteratively fill holes
            for _ in range(10):  # Max iterations
                # Dilate valid pixels
                valid_mask = ~invalid_mask
                dilated_valid = cv2.dilate(valid_mask.astype(np.uint8), 
                                         np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                
                # Find newly valid pixels
                newly_valid = dilated_valid & invalid_mask
                
                if not np.any(newly_valid):
                    break
                
                # Interpolate values for newly valid pixels
                smoothed = cv2.filter2D(filled_depth, -1, kernel)
                filled_depth[newly_valid] = smoothed[newly_valid]
                invalid_mask[newly_valid] = False
        
        return filled_depth
    
    def smooth_depth(
        self,
        depth: np.ndarray,
        method: str = "bilateral",
        kernel_size: int = 5,
        sigma_color: float = 50.0,
        sigma_space: float = 50.0
    ) -> np.ndarray:
        """Smooth depth map while preserving edges.
        
        Args:
            depth: Input depth map.
            method: Smoothing method ('bilateral', 'gaussian', 'median').
            kernel_size: Kernel size for filtering.
            sigma_color: Filter sigma in the color space (bilateral only).
            sigma_space: Filter sigma in the coordinate space (bilateral only).
            
        Returns:
            Smoothed depth map.
        """
        # Create mask for valid pixels
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return depth.copy()
        
        smoothed_depth = depth.copy()
        
        if method == "bilateral":
            # Convert to float32 for bilateral filter (OpenCV requirement)
            depth_float32 = depth.astype(np.float32)
            smoothed_depth = cv2.bilateralFilter(depth_float32, kernel_size, sigma_color, sigma_space)
            
        elif method == "gaussian":
            smoothed_depth = cv2.GaussianBlur(depth, (kernel_size, kernel_size), 0)
            
        elif method == "median":
            smoothed_depth = cv2.medianBlur(depth.astype(np.float32), kernel_size)
        
        # Preserve invalid pixels
        smoothed_depth[~valid_mask] = depth[~valid_mask]
        
        return smoothed_depth
    
    def process_depth_map(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        fill_holes: bool = True,
        smooth: bool = True
    ) -> np.ndarray:
        """Apply complete post-processing pipeline to depth map.
        
        Args:
            depth: Input depth map.
            image: Optional RGB image for guided processing.
            fill_holes: Whether to fill holes.
            smooth: Whether to apply smoothing.
            
        Returns:
            Post-processed depth map.
        """
        processed_depth = depth.copy()
        
        logger.info("Starting depth map post-processing")
        
        # Fill holes
        if fill_holes:
            logger.debug("Filling holes")
            processed_depth = self.fill_holes(processed_depth)
        
        # Apply smoothing
        if smooth:
            logger.debug("Applying smoothing")
            processed_depth = self.smooth_depth(processed_depth)
        
        logger.info("Depth map post-processing completed")
        
        return processed_depth 