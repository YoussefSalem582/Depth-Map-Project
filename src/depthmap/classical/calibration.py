"""Stereo camera calibration utilities."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class StereoCalibrator:
    """Stereo camera calibration using checkerboard patterns."""
    
    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 1.0,
        flags: int = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ):
        """Initialize stereo calibrator.
        
        Args:
            checkerboard_size: Number of inner corners (width, height).
            square_size: Size of checkerboard squares in real-world units.
            flags: OpenCV calibration flags.
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.flags = flags
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration data
        self.objpoints: List[np.ndarray] = []  # 3D points in real world space
        self.imgpoints_left: List[np.ndarray] = []  # 2D points in left image plane
        self.imgpoints_right: List[np.ndarray] = []  # 2D points in right image plane
        self.image_size: Optional[Tuple[int, int]] = None
        
        # Calibration results
        self.camera_matrix_left: Optional[np.ndarray] = None
        self.dist_coeffs_left: Optional[np.ndarray] = None
        self.camera_matrix_right: Optional[np.ndarray] = None
        self.dist_coeffs_right: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None  # Rotation matrix
        self.T: Optional[np.ndarray] = None  # Translation vector
        self.E: Optional[np.ndarray] = None  # Essential matrix
        self.F: Optional[np.ndarray] = None  # Fundamental matrix
        self.reprojection_error: Optional[float] = None
    
    def add_image_pair(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        visualize: bool = False
    ) -> bool:
        """Add a stereo image pair for calibration.
        
        Args:
            img_left: Left camera image.
            img_right: Right camera image.
            visualize: Whether to show detected corners.
            
        Returns:
            True if corners were found in both images, False otherwise.
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
        
        # Store image size
        if self.image_size is None:
            self.image_size = gray_left.shape[::-1]  # (width, height)
        
        # Find checkerboard corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, self.checkerboard_size, self.flags
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, self.checkerboard_size, self.flags
        )
        
        if ret_left and ret_right:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            # Store points
            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            
            logger.info(f"Added calibration image pair. Total pairs: {len(self.objpoints)}")
            
            # Visualize if requested
            if visualize:
                img_left_vis = img_left.copy()
                img_right_vis = img_right.copy()
                cv2.drawChessboardCorners(img_left_vis, self.checkerboard_size, corners_left, ret_left)
                cv2.drawChessboardCorners(img_right_vis, self.checkerboard_size, corners_right, ret_right)
                
                # Show images side by side
                combined = np.hstack([img_left_vis, img_right_vis])
                cv2.imshow('Detected Corners', combined)
                cv2.waitKey(500)
            
            return True
        else:
            logger.warning("Could not find checkerboard corners in both images")
            return False
    
    def calibrate(self) -> bool:
        """Perform stereo calibration.
        
        Returns:
            True if calibration was successful, False otherwise.
        """
        if len(self.objpoints) < 10:
            logger.error(f"Need at least 10 image pairs for calibration, got {len(self.objpoints)}")
            return False
        
        if self.image_size is None:
            logger.error("No images have been added for calibration")
            return False
        
        logger.info(f"Starting stereo calibration with {len(self.objpoints)} image pairs")
        
        # Individual camera calibration
        logger.info("Calibrating left camera...")
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, self.image_size, None, None
        )
        
        logger.info("Calibrating right camera...")
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, self.image_size, None, None
        )
        
        # Stereo calibration
        logger.info("Performing stereo calibration...")
        stereo_flags = (cv2.CALIB_FIX_INTRINSIC + 
                       cv2.CALIB_RATIONAL_MODEL + 
                       cv2.CALIB_FIX_PRINCIPAL_POINT)
        
        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            mtx_left, dist_left,
            mtx_right, dist_right,
            self.image_size,
            flags=stereo_flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        )
        
        if ret_stereo:
            self.camera_matrix_left = mtx_left
            self.dist_coeffs_left = dist_left
            self.camera_matrix_right = mtx_right
            self.dist_coeffs_right = dist_right
            self.R = R
            self.T = T
            self.E = E
            self.F = F
            self.reprojection_error = ret_stereo
            
            logger.info(f"Stereo calibration successful! Reprojection error: {ret_stereo:.4f}")
            logger.info(f"Baseline: {np.linalg.norm(T):.4f} units")
            
            return True
        else:
            logger.error("Stereo calibration failed")
            return False
    
    def get_calibration_data(self) -> dict:
        """Get calibration data as dictionary.
        
        Returns:
            Dictionary containing all calibration parameters.
        """
        if self.camera_matrix_left is None:
            raise ValueError("Calibration has not been performed yet")
        
        return {
            "image_size": self.image_size,
            "camera_matrix_left": self.camera_matrix_left.tolist(),
            "dist_coeffs_left": self.dist_coeffs_left.tolist(),
            "camera_matrix_right": self.camera_matrix_right.tolist(),
            "dist_coeffs_right": self.dist_coeffs_right.tolist(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "reprojection_error": float(self.reprojection_error),
            "checkerboard_size": self.checkerboard_size,
            "square_size": self.square_size,
        }


def save_calibration(calibration_data: dict, output_path: Union[str, Path]) -> None:
    """Save calibration data to YAML file.
    
    Args:
        calibration_data: Calibration data dictionary.
        output_path: Path to save the calibration file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    logger.info(f"Calibration data saved to {output_path}")


def load_calibration(calibration_path: Union[str, Path]) -> dict:
    """Load calibration data from YAML file.
    
    Args:
        calibration_path: Path to the calibration file.
        
    Returns:
        Calibration data dictionary.
        
    Raises:
        FileNotFoundError: If calibration file doesn't exist.
    """
    calibration_path = Path(calibration_path)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    
    with open(calibration_path, 'r') as f:
        calibration_data = yaml.safe_load(f)
    
    # Convert lists back to numpy arrays
    for key in ["camera_matrix_left", "dist_coeffs_left", "camera_matrix_right", 
                "dist_coeffs_right", "R", "T", "E", "F"]:
        if key in calibration_data:
            calibration_data[key] = np.array(calibration_data[key])
    
    logger.info(f"Calibration data loaded from {calibration_path}")
    return calibration_data


def calibrate_from_images(
    left_images: List[Union[str, Path]],
    right_images: List[Union[str, Path]],
    checkerboard_size: Tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    output_path: Optional[Union[str, Path]] = None
) -> dict:
    """Calibrate stereo cameras from image lists.
    
    Args:
        left_images: List of paths to left camera images.
        right_images: List of paths to right camera images.
        checkerboard_size: Number of inner corners (width, height).
        square_size: Size of checkerboard squares in real-world units.
        output_path: Optional path to save calibration data.
        
    Returns:
        Calibration data dictionary.
        
    Raises:
        ValueError: If image lists have different lengths or calibration fails.
    """
    if len(left_images) != len(right_images):
        raise ValueError("Left and right image lists must have the same length")
    
    calibrator = StereoCalibrator(checkerboard_size, square_size)
    
    successful_pairs = 0
    for left_path, right_path in zip(left_images, right_images):
        logger.info(f"Processing image pair: {left_path}, {right_path}")
        
        img_left = cv2.imread(str(left_path))
        img_right = cv2.imread(str(right_path))
        
        if img_left is None or img_right is None:
            logger.warning(f"Could not load image pair: {left_path}, {right_path}")
            continue
        
        if calibrator.add_image_pair(img_left, img_right):
            successful_pairs += 1
    
    logger.info(f"Successfully processed {successful_pairs} image pairs")
    
    if not calibrator.calibrate():
        raise ValueError("Stereo calibration failed")
    
    calibration_data = calibrator.get_calibration_data()
    
    if output_path is not None:
        save_calibration(calibration_data, output_path)
    
    return calibration_data 