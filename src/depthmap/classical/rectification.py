"""Stereo image rectification utilities."""

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StereoRectifier:
    """Stereo image rectification for parallel camera geometry."""
    
    def __init__(
        self,
        camera_matrix_left: np.ndarray,
        dist_coeffs_left: np.ndarray,
        camera_matrix_right: np.ndarray,
        dist_coeffs_right: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        image_size: Tuple[int, int],
        alpha: float = 0.0
    ):
        """Initialize stereo rectifier.
        
        Args:
            camera_matrix_left: Left camera matrix (3x3).
            dist_coeffs_left: Left camera distortion coefficients.
            camera_matrix_right: Right camera matrix (3x3).
            dist_coeffs_right: Right camera distortion coefficients.
            R: Rotation matrix between cameras (3x3).
            T: Translation vector between cameras (3x1).
            image_size: Image size as (width, height).
            alpha: Free scaling parameter (0=no black pixels, 1=all pixels retained).
        """
        self.camera_matrix_left = camera_matrix_left
        self.dist_coeffs_left = dist_coeffs_left
        self.camera_matrix_right = camera_matrix_right
        self.dist_coeffs_right = dist_coeffs_right
        self.R = R
        self.T = T
        self.image_size = image_size
        self.alpha = alpha
        
        # Compute rectification transforms
        self._compute_rectification()
        
        logger.info("Stereo rectifier initialized")
    
    def _compute_rectification(self) -> None:
        """Compute stereo rectification transforms."""
        # Stereo rectification
        (self.R1, self.R2, self.P1, self.P2, self.Q, 
         self.valid_roi_left, self.valid_roi_right) = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            self.image_size, self.R, self.T,
            alpha=self.alpha
        )
        
        # Compute rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1,
            self.image_size, cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2,
            self.image_size, cv2.CV_16SC2
        )
        
        logger.debug("Rectification transforms computed")
    
    def rectify_images(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify stereo image pair.
        
        Args:
            img_left: Left camera image.
            img_right: Right camera image.
            interpolation: Interpolation method for remapping.
            
        Returns:
            Tuple of (rectified_left, rectified_right) images.
        """
        # Apply rectification maps
        rect_left = cv2.remap(
            img_left, self.map1_left, self.map2_left, interpolation
        )
        rect_right = cv2.remap(
            img_right, self.map1_right, self.map2_right, interpolation
        )
        
        return rect_left, rect_right
    
    def rectify_points(
        self,
        points_left: np.ndarray,
        points_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify corresponding points in stereo images.
        
        Args:
            points_left: Points in left image (Nx2).
            points_right: Points in right image (Nx2).
            
        Returns:
            Tuple of (rectified_points_left, rectified_points_right).
        """
        # Undistort and rectify points
        rect_points_left = cv2.undistortPoints(
            points_left.reshape(-1, 1, 2),
            self.camera_matrix_left, self.dist_coeffs_left,
            R=self.R1, P=self.P1
        ).reshape(-1, 2)
        
        rect_points_right = cv2.undistortPoints(
            points_right.reshape(-1, 1, 2),
            self.camera_matrix_right, self.dist_coeffs_right,
            R=self.R2, P=self.P2
        ).reshape(-1, 2)
        
        return rect_points_left, rect_points_right
    
    def get_rectified_camera_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get rectified camera projection matrices.
        
        Returns:
            Tuple of (P1, P2) rectified projection matrices.
        """
        return self.P1, self.P2
    
    def get_disparity_to_depth_matrix(self) -> np.ndarray:
        """Get disparity-to-depth reprojection matrix.
        
        Returns:
            4x4 reprojection matrix Q.
        """
        return self.Q
    
    def get_valid_rois(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Get valid regions of interest after rectification.
        
        Returns:
            Tuple of (left_roi, right_roi) as (x, y, width, height).
        """
        return self.valid_roi_left, self.valid_roi_right
    
    def crop_to_valid_roi(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop rectified images to valid regions of interest.
        
        Args:
            img_left: Rectified left image.
            img_right: Rectified right image.
            
        Returns:
            Tuple of (cropped_left, cropped_right) images.
        """
        x1, y1, w1, h1 = self.valid_roi_left
        x2, y2, w2, h2 = self.valid_roi_right
        
        # Use intersection of both ROIs
        x = max(x1, x2)
        y = max(y1, y2)
        w = min(x1 + w1, x2 + w2) - x
        h = min(y1 + h1, y2 + h2) - y
        
        if w <= 0 or h <= 0:
            logger.warning("No valid intersection of ROIs")
            return img_left, img_right
        
        cropped_left = img_left[y:y+h, x:x+w]
        cropped_right = img_right[y:y+h, x:x+w]
        
        return cropped_left, cropped_right
    
    def visualize_rectification(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        line_spacing: int = 50
    ) -> np.ndarray:
        """Visualize rectification quality with horizontal lines.
        
        Args:
            img_left: Rectified left image.
            img_right: Rectified right image.
            line_spacing: Spacing between horizontal lines in pixels.
            
        Returns:
            Side-by-side visualization with horizontal lines.
        """
        # Ensure images are the same size
        h, w = img_left.shape[:2]
        if img_right.shape[:2] != (h, w):
            img_right = cv2.resize(img_right, (w, h))
        
        # Convert to color if grayscale
        if len(img_left.shape) == 2:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        if len(img_right.shape) == 2:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side image
        combined = np.hstack([img_left, img_right])
        
        # Draw horizontal lines
        for y in range(0, h, line_spacing):
            cv2.line(combined, (0, y), (2*w, y), (0, 255, 0), 1)
        
        # Draw vertical separator
        cv2.line(combined, (w, 0), (w, h), (255, 0, 0), 2)
        
        return combined


def rectify_stereo_pair(
    img_left: np.ndarray,
    img_right: np.ndarray,
    calibration_data: dict,
    alpha: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, StereoRectifier]:
    """Convenience function to rectify a stereo pair using calibration data.
    
    Args:
        img_left: Left camera image.
        img_right: Right camera image.
        calibration_data: Dictionary containing calibration parameters.
        alpha: Free scaling parameter for rectification.
        
    Returns:
        Tuple of (rectified_left, rectified_right, rectifier).
    """
    # Extract calibration parameters
    camera_matrix_left = calibration_data["camera_matrix_left"]
    dist_coeffs_left = calibration_data["dist_coeffs_left"]
    camera_matrix_right = calibration_data["camera_matrix_right"]
    dist_coeffs_right = calibration_data["dist_coeffs_right"]
    R = calibration_data["R"]
    T = calibration_data["T"]
    image_size = calibration_data["image_size"]
    
    # Create rectifier
    rectifier = StereoRectifier(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        R, T, image_size, alpha
    )
    
    # Rectify images
    rect_left, rect_right = rectifier.rectify_images(img_left, img_right)
    
    return rect_left, rect_right, rectifier 