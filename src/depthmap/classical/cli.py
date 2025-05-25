"""Command line interface for classical stereo depth estimation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .calibration import StereoCalibrator, load_calibration, save_calibration, calibrate_from_images
from .stereo_depth import StereoDepthEstimator
from ..utils.io import load_image, save_image, save_depth
from ..utils.logging import setup_logger
from ..utils.visualization import colorize_depth

logger = logging.getLogger(__name__)


def estimate_depth_command(args) -> None:
    """Estimate depth from stereo image pair."""
    logger.info(f"Loading stereo images: {args.left}, {args.right}")
    
    # Load images
    try:
        img_left = load_image(args.left, color_mode="RGB")
        img_right = load_image(args.right, color_mode="RGB")
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        sys.exit(1)
    
    # Load calibration if provided
    focal_length = args.focal_length
    baseline = args.baseline
    
    if args.calib:
        try:
            calib_data = load_calibration(args.calib)
            focal_length = calib_data["camera_matrix_left"][0, 0]
            baseline = np.linalg.norm(calib_data["T"])
            logger.info(f"Loaded calibration: focal_length={focal_length:.2f}, baseline={baseline:.4f}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            sys.exit(1)
    
    # Initialize estimator
    estimator = StereoDepthEstimator(
        method=args.method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        min_disparity=args.min_disparity,
        uniqueness_ratio=args.uniqueness_ratio,
        speckle_window_size=args.speckle_window_size,
        speckle_range=args.speckle_range
    )
    
    # Estimate depth
    logger.info("Estimating depth...")
    try:
        depth, disparity = estimator.estimate_depth(
            img_left, img_right,
            focal_length=focal_length,
            baseline=baseline,
            min_depth=args.min_depth,
            max_depth=args.max_depth
        )
    except Exception as e:
        logger.error(f"Depth estimation failed: {e}")
        sys.exit(1)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save depth map
    logger.info(f"Saving depth map to {output_path}")
    save_depth(depth, output_path, scale_factor=args.depth_scale)
    
    # Save disparity if requested
    if args.save_disparity:
        disp_path = output_path.with_suffix(".disp.png")
        logger.info(f"Saving disparity map to {disp_path}")
        save_depth(disparity, disp_path)
    
    # Save visualization if requested
    if args.save_visualization:
        vis_path = output_path.with_suffix(".vis.png")
        logger.info(f"Saving visualization to {vis_path}")
        depth_colored = colorize_depth(depth, colormap=args.colormap)
        save_image(depth_colored, vis_path, color_mode="RGB")
    
    logger.info("Depth estimation completed successfully!")


def calibrate_command(args) -> None:
    """Calibrate stereo cameras from checkerboard images."""
    logger.info("Starting stereo calibration...")
    
    # Get image lists
    left_images = sorted(Path(args.left_images).glob("*.png"))
    right_images = sorted(Path(args.right_images).glob("*.png"))
    
    if len(left_images) == 0 or len(right_images) == 0:
        logger.error("No images found in specified directories")
        sys.exit(1)
    
    if len(left_images) != len(right_images):
        logger.error(f"Mismatch in image count: {len(left_images)} left, {len(right_images)} right")
        sys.exit(1)
    
    logger.info(f"Found {len(left_images)} image pairs")
    
    # Perform calibration
    try:
        calibration_data = calibrate_from_images(
            left_images=left_images,
            right_images=right_images,
            checkerboard_size=(args.checkerboard_width, args.checkerboard_height),
            square_size=args.square_size,
            output_path=args.output
        )
        
        logger.info(f"Calibration successful! Reprojection error: {calibration_data['reprojection_error']:.4f}")
        logger.info(f"Baseline: {np.linalg.norm(calibration_data['T']):.4f} units")
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classical stereo depth estimation and calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Depth estimation command
    depth_parser = subparsers.add_parser(
        "estimate",
        help="Estimate depth from stereo image pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    depth_parser.add_argument(
        "--left", "-l",
        required=True,
        help="Path to left camera image"
    )
    
    depth_parser.add_argument(
        "--right", "-r",
        required=True,
        help="Path to right camera image"
    )
    
    depth_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for depth map"
    )
    
    depth_parser.add_argument(
        "--calib", "-c",
        help="Path to calibration file (YAML)"
    )
    
    depth_parser.add_argument(
        "--method",
        choices=["BM", "SGBM"],
        default="SGBM",
        help="Stereo matching method"
    )
    
    depth_parser.add_argument(
        "--focal-length",
        type=float,
        default=721.5377,
        help="Camera focal length in pixels (used if no calibration file)"
    )
    
    depth_parser.add_argument(
        "--baseline",
        type=float,
        default=0.54,
        help="Stereo baseline in meters (used if no calibration file)"
    )
    
    depth_parser.add_argument(
        "--num-disparities",
        type=int,
        default=96,
        help="Maximum disparity (must be divisible by 16)"
    )
    
    depth_parser.add_argument(
        "--block-size",
        type=int,
        default=11,
        help="Matched block size (must be odd)"
    )
    
    depth_parser.add_argument(
        "--min-disparity",
        type=int,
        default=0,
        help="Minimum disparity"
    )
    
    depth_parser.add_argument(
        "--uniqueness-ratio",
        type=int,
        default=10,
        help="Uniqueness ratio for disparity validation"
    )
    
    depth_parser.add_argument(
        "--speckle-window-size",
        type=int,
        default=100,
        help="Speckle filter window size"
    )
    
    depth_parser.add_argument(
        "--speckle-range",
        type=int,
        default=32,
        help="Speckle filter range"
    )
    
    depth_parser.add_argument(
        "--min-depth",
        type=float,
        default=0.1,
        help="Minimum depth in meters"
    )
    
    depth_parser.add_argument(
        "--max-depth",
        type=float,
        default=100.0,
        help="Maximum depth in meters"
    )
    
    depth_parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for saving depth (e.g., 1000 for mm)"
    )
    
    depth_parser.add_argument(
        "--save-disparity",
        action="store_true",
        help="Save disparity map"
    )
    
    depth_parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save colored depth visualization"
    )
    
    depth_parser.add_argument(
        "--colormap",
        default="turbo",
        help="Colormap for visualization"
    )
    
    # Calibration command
    calib_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate stereo cameras from checkerboard images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    calib_parser.add_argument(
        "--left-images",
        required=True,
        help="Directory containing left camera calibration images"
    )
    
    calib_parser.add_argument(
        "--right-images",
        required=True,
        help="Directory containing right camera calibration images"
    )
    
    calib_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for calibration file (YAML)"
    )
    
    calib_parser.add_argument(
        "--checkerboard-width",
        type=int,
        default=9,
        help="Number of inner corners in checkerboard width"
    )
    
    calib_parser.add_argument(
        "--checkerboard-height",
        type=int,
        default=6,
        help="Number of inner corners in checkerboard height"
    )
    
    calib_parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Size of checkerboard squares in real-world units"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger("depthmap", level=log_level)
    
    # Execute command
    if args.command == "estimate":
        estimate_depth_command(args)
    elif args.command == "calibrate":
        calibrate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 