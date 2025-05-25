"""Depth map post-processing utilities."""

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import filters, morphology, segmentation

logger = logging.getLogger(__name__)


class DepthPostProcessor:
    """Advanced post-processing utilities for depth maps."""
    
    def __init__(self):
        """Initialize depth post-processor."""
        logger.info("Advanced depth post-processor initialized")
    
    def fill_holes(
        self,
        depth: np.ndarray,
        method: str = "inpaint",
        kernel_size: int = 5,
        max_hole_size: int = 100
    ) -> np.ndarray:
        """Fill holes in depth map with advanced techniques.
        
        Args:
            depth: Input depth map.
            method: Hole filling method ('inpaint', 'interpolate', 'median', 'morphological').
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
            # Use OpenCV inpainting with improved parameters
            depth_uint8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask_uint8 = invalid_mask.astype(np.uint8) * 255
            
            # Use both inpainting methods and blend
            inpainted_telea = cv2.inpaint(depth_uint8, mask_uint8, 5, cv2.INPAINT_TELEA)
            inpainted_ns = cv2.inpaint(depth_uint8, mask_uint8, 5, cv2.INPAINT_NS)
            
            # Blend the two methods
            inpainted = cv2.addWeighted(inpainted_telea, 0.6, inpainted_ns, 0.4, 0)
            
            # Convert back to original scale
            depth_min, depth_max = depth[~invalid_mask].min(), depth[~invalid_mask].max()
            filled_depth[invalid_mask] = (inpainted[invalid_mask] / 255.0) * (depth_max - depth_min) + depth_min
            
        elif method == "morphological":
            # Advanced morphological hole filling
            filled_depth = self._morphological_fill_holes(depth, invalid_mask, max_hole_size)
            
        elif method == "interpolate":
            # Improved interpolation using valid neighbors
            filled_depth = self._advanced_interpolation(depth, invalid_mask, kernel_size)
        
        return filled_depth
    
    def _morphological_fill_holes(self, depth: np.ndarray, invalid_mask: np.ndarray, max_hole_size: int) -> np.ndarray:
        """Fill holes using morphological operations."""
        filled_depth = depth.copy()
        
        # Find connected components of holes
        labeled_holes, num_holes = ndimage.label(invalid_mask)
        
        for hole_id in range(1, num_holes + 1):
            hole_mask = labeled_holes == hole_id
            hole_size = np.sum(hole_mask)
            
            if hole_size <= max_hole_size:
                # Dilate the hole boundary to get surrounding valid pixels
                dilated = binary_dilation(hole_mask, iterations=3)
                boundary = dilated & ~hole_mask & ~invalid_mask
                
                if np.any(boundary):
                    # Use median of boundary pixels
                    boundary_values = depth[boundary]
                    fill_value = np.median(boundary_values)
                    filled_depth[hole_mask] = fill_value
        
        return filled_depth
    
    def _advanced_interpolation(self, depth: np.ndarray, invalid_mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Advanced interpolation with distance weighting."""
        filled_depth = depth.copy()
        
        # Create distance transform from valid pixels
        valid_mask = ~invalid_mask
        distance_transform = ndimage.distance_transform_edt(~valid_mask)
        
        # Iteratively fill holes
        for iteration in range(10):
            # Find pixels that can be filled (adjacent to valid pixels)
            dilated_valid = binary_dilation(valid_mask, iterations=1)
            newly_fillable = dilated_valid & invalid_mask
            
            if not np.any(newly_fillable):
                break
            
            # For each newly fillable pixel, interpolate from neighbors
            for y, x in np.argwhere(newly_fillable):
                # Get neighborhood
                y_start, y_end = max(0, y - kernel_size//2), min(depth.shape[0], y + kernel_size//2 + 1)
                x_start, x_end = max(0, x - kernel_size//2), min(depth.shape[1], x + kernel_size//2 + 1)
                
                neighborhood = depth[y_start:y_end, x_start:x_end]
                neighborhood_valid = valid_mask[y_start:y_end, x_start:x_end]
                
                if np.any(neighborhood_valid):
                    # Distance-weighted interpolation
                    valid_values = neighborhood[neighborhood_valid]
                    filled_depth[y, x] = np.mean(valid_values)
                    valid_mask[y, x] = True
                    invalid_mask[y, x] = False
        
        return filled_depth

    def smooth_depth(
        self,
        depth: np.ndarray,
        method: str = "bilateral",
        kernel_size: int = 5,
        sigma_color: float = 50.0,
        sigma_space: float = 50.0,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Advanced depth smoothing while preserving edges.
        
        Args:
            depth: Input depth map.
            method: Smoothing method ('bilateral', 'guided', 'edge_preserving', 'anisotropic').
            kernel_size: Kernel size for filtering.
            sigma_color: Filter sigma in the color space.
            sigma_space: Filter sigma in the coordinate space.
            image: Optional RGB image for guided filtering.
            
        Returns:
            Smoothed depth map.
        """
        # Create mask for valid pixels
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return depth.copy()
        
        smoothed_depth = depth.copy()
        
        if method == "bilateral":
            # Enhanced bilateral filtering
            depth_float32 = depth.astype(np.float32)
            smoothed_depth = cv2.bilateralFilter(depth_float32, kernel_size, sigma_color, sigma_space)
            
        elif method == "guided" and image is not None:
            # Guided filter using RGB image
            smoothed_depth = self._guided_filter(depth, image, radius=kernel_size, epsilon=0.01)
            
        elif method == "edge_preserving":
            # Edge-preserving filter
            depth_uint8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            smoothed_uint8 = cv2.edgePreservingFilter(depth_uint8, flags=2, sigma_s=50, sigma_r=0.4)
            
            # Convert back to original scale
            depth_min, depth_max = depth[valid_mask].min(), depth[valid_mask].max()
            smoothed_depth = (smoothed_uint8.astype(np.float32) / 255.0) * (depth_max - depth_min) + depth_min
            
        elif method == "anisotropic":
            # Anisotropic diffusion
            smoothed_depth = self._anisotropic_diffusion(depth, num_iter=10, kappa=50, gamma=0.1)
        
        # Preserve invalid pixels
        smoothed_depth[~valid_mask] = depth[~valid_mask]
        
        return smoothed_depth
    
    def _guided_filter(self, depth: np.ndarray, guide: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
        """Guided filter implementation."""
        # Convert guide to grayscale if needed
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        
        guide = guide.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)
        
        # Box filter
        def box_filter(img, r):
            return cv2.boxFilter(img, -1, (2*r+1, 2*r+1))
        
        N = box_filter(np.ones_like(guide), radius)
        
        mean_I = box_filter(guide, radius) / N
        mean_p = box_filter(depth, radius) / N
        mean_Ip = box_filter(guide * depth, radius) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = box_filter(guide * guide, radius) / N
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        
        mean_a = box_filter(a, radius) / N
        mean_b = box_filter(b, radius) / N
        
        return mean_a * guide + mean_b
    
    def _anisotropic_diffusion(self, depth: np.ndarray, num_iter: int, kappa: float, gamma: float) -> np.ndarray:
        """Anisotropic diffusion for edge-preserving smoothing."""
        img = depth.astype(np.float32)
        
        for _ in range(num_iter):
            # Calculate gradients
            nabla_N = np.roll(img, -1, axis=0) - img
            nabla_S = np.roll(img, 1, axis=0) - img
            nabla_E = np.roll(img, -1, axis=1) - img
            nabla_W = np.roll(img, 1, axis=1) - img
            
            # Calculate diffusion coefficients
            cN = np.exp(-(nabla_N / kappa) ** 2)
            cS = np.exp(-(nabla_S / kappa) ** 2)
            cE = np.exp(-(nabla_E / kappa) ** 2)
            cW = np.exp(-(nabla_W / kappa) ** 2)
            
            # Update image
            img += gamma * (cN * nabla_N + cS * nabla_S + cE * nabla_E + cW * nabla_W)
        
        return img
    
    def enhance_depth_edges(self, depth: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        """Enhance depth map edges using image guidance."""
        if image is None:
            # Use depth-based edge enhancement
            edges = cv2.Canny((depth * 255).astype(np.uint8), 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            
            # Sharpen edges
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(depth, -1, kernel)
            
            # Blend based on edge mask
            edge_mask = edges.astype(np.float32) / 255.0
            enhanced = depth * (1 - edge_mask) + sharpened * edge_mask
            
        else:
            # Use image edges to guide depth enhancement
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_edges = cv2.Canny(gray, 50, 150)
            image_edges = cv2.dilate(image_edges, np.ones((3, 3), np.uint8), iterations=1)
            
            # Apply guided filtering at edges
            enhanced = self._guided_filter(depth, image, radius=3, epsilon=0.001)
            
            # Blend with original based on edge strength
            edge_mask = image_edges.astype(np.float32) / 255.0
            enhanced = depth * (1 - edge_mask * 0.5) + enhanced * (edge_mask * 0.5)
        
        return enhanced
    
    def multi_scale_refinement(self, depth: np.ndarray, scales: list = [0.5, 1.0, 2.0]) -> np.ndarray:
        """Multi-scale depth refinement."""
        original_shape = depth.shape
        refined_depths = []
        
        for scale in scales:
            if scale != 1.0:
                # Resize depth map
                new_size = (int(original_shape[1] * scale), int(original_shape[0] * scale))
                scaled_depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_LINEAR)
            else:
                scaled_depth = depth.copy()
            
            # Apply processing at this scale
            processed = self.smooth_depth(scaled_depth, method="bilateral")
            processed = self.fill_holes(processed, method="morphological")
            
            # Resize back to original size
            if scale != 1.0:
                processed = cv2.resize(processed, (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            
            refined_depths.append(processed)
        
        # Weighted fusion of multi-scale results
        weights = [0.2, 0.6, 0.2]  # Give more weight to original scale
        fused_depth = np.zeros_like(depth)
        
        for depth_map, weight in zip(refined_depths, weights):
            fused_depth += depth_map * weight
        
        return fused_depth

    def process_depth_map(
        self,
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        fill_holes: bool = True,
        smooth: bool = True,
        enhance_edges: bool = True,
        multi_scale: bool = True
    ) -> np.ndarray:
        """Apply complete advanced post-processing pipeline to depth map.
        
        Args:
            depth: Input depth map.
            image: Optional RGB image for guided processing.
            fill_holes: Whether to fill holes.
            smooth: Whether to apply smoothing.
            enhance_edges: Whether to enhance edges.
            multi_scale: Whether to apply multi-scale refinement.
            
        Returns:
            Post-processed depth map.
        """
        processed_depth = depth.copy()
        
        logger.info("Starting advanced depth map post-processing")
        
        # Multi-scale refinement
        if multi_scale:
            logger.debug("Applying multi-scale refinement")
            processed_depth = self.multi_scale_refinement(processed_depth)
        
        # Fill holes with advanced method
        if fill_holes:
            logger.debug("Filling holes with morphological method")
            processed_depth = self.fill_holes(processed_depth, method="morphological")
        
        # Apply advanced smoothing
        if smooth:
            if image is not None:
                logger.debug("Applying guided smoothing")
                processed_depth = self.smooth_depth(processed_depth, method="guided", image=image)
            else:
                logger.debug("Applying edge-preserving smoothing")
                processed_depth = self.smooth_depth(processed_depth, method="edge_preserving")
        
        # Enhance edges
        if enhance_edges:
            logger.debug("Enhancing depth edges")
            processed_depth = self.enhance_depth_edges(processed_depth, image)
        
        logger.info("Advanced depth map post-processing completed")
        
        return processed_depth 