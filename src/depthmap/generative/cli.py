"""Command line interface for generative depth estimation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .midas import MiDaSDepthEstimator
from ..utils.io import load_image, save_image, save_depth
from ..utils.logging import setup_logger
from ..utils.visualization import colorize_depth

logger = logging.getLogger(__name__)


def predict_command(args) -> None:
    """Predict depth from single image using generative model."""
    logger.info(f"Loading image: {args.input}")
    
    # Load image
    try:
        image = load_image(args.input, color_mode="RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)
    
    # Initialize estimator
    logger.info(f"Initializing {args.model} depth estimator")
    try:
        estimator = MiDaSDepthEstimator(
            model_name=args.model,
            device=args.device,
            enable_amp=args.enable_amp
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Predict depth
    logger.info("Predicting depth...")
    try:
        if args.multiscale:
            depth = estimator.predict_multiscale(
                image,
                scales=args.scales,
                fusion_method=args.fusion_method
            )
        else:
            depth = estimator.predict(image)
    except Exception as e:
        logger.error(f"Depth prediction failed: {e}")
        sys.exit(1)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save depth map
    logger.info(f"Saving depth map to {output_path}")
    save_depth(depth, output_path, scale_factor=args.depth_scale)
    
    # Save visualization if requested
    if args.save_visualization:
        vis_path = output_path.with_suffix(".vis.png")
        logger.info(f"Saving visualization to {vis_path}")
        depth_colored = colorize_depth(depth, colormap=args.colormap)
        save_image(depth_colored, vis_path, color_mode="RGB")
    
    # Save confidence map if requested
    if args.save_confidence:
        conf_path = output_path.with_suffix(".conf.png")
        logger.info(f"Saving confidence map to {conf_path}")
        depth_pred, confidence = estimator.predict_with_confidence(image, return_confidence=True)
        save_depth(confidence, conf_path)
    
    logger.info("Depth prediction completed successfully!")


def batch_predict_command(args) -> None:
    """Predict depth for batch of images."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Get image files
    image_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        logger.error(f"No images found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Initialize estimator
    logger.info(f"Initializing {args.model} depth estimator")
    try:
        estimator = MiDaSDepthEstimator(
            model_name=args.model,
            device=args.device,
            enable_amp=args.enable_amp
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    for i, image_file in enumerate(image_files):
        logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        try:
            # Load image
            image = load_image(image_file, color_mode="RGB")
            
            # Predict depth
            if args.multiscale:
                depth = estimator.predict_multiscale(
                    image,
                    scales=args.scales,
                    fusion_method=args.fusion_method
                )
            else:
                depth = estimator.predict(image)
            
            # Save depth map
            output_path = output_dir / f"{image_file.stem}_depth.png"
            save_depth(depth, output_path, scale_factor=args.depth_scale)
            
            # Save visualization if requested
            if args.save_visualization:
                vis_path = output_dir / f"{image_file.stem}_depth_vis.png"
                depth_colored = colorize_depth(depth, colormap=args.colormap)
                save_image(depth_colored, vis_path, color_mode="RGB")
                
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            continue
    
    logger.info("Batch processing completed!")


def benchmark_command(args) -> None:
    """Benchmark model inference speed."""
    logger.info(f"Loading test image: {args.input}")
    
    # Load image
    try:
        image = load_image(args.input, color_mode="RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)
    
    # Initialize estimator
    logger.info(f"Initializing {args.model} depth estimator")
    try:
        estimator = MiDaSDepthEstimator(
            model_name=args.model,
            device=args.device,
            enable_amp=args.enable_amp
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Run benchmark
    logger.info(f"Running benchmark with {args.num_runs} runs...")
    try:
        results = estimator.benchmark_inference(
            image,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )
        
        logger.info("Benchmark Results:")
        logger.info(f"  Mean time: {results['mean_time']:.4f}s")
        logger.info(f"  Std time: {results['std_time']:.4f}s")
        logger.info(f"  Min time: {results['min_time']:.4f}s")
        logger.info(f"  Max time: {results['max_time']:.4f}s")
        logger.info(f"  FPS: {results['fps']:.2f}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generative depth estimation using MiDaS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict depth from single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    predict_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input image"
    )
    
    predict_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for depth map"
    )
    
    predict_parser.add_argument(
        "--model",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS", "MiDaS_small"],
        default="DPT_Large",
        help="MiDaS model to use"
    )
    
    predict_parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on (auto, cpu, cuda, mps)"
    )
    
    predict_parser.add_argument(
        "--enable-amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision"
    )
    
    predict_parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Use multi-scale prediction"
    )
    
    predict_parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5],
        help="Scales for multi-scale prediction"
    )
    
    predict_parser.add_argument(
        "--fusion-method",
        choices=["average", "median", "max"],
        default="average",
        help="Fusion method for multi-scale prediction"
    )
    
    predict_parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for saving depth (e.g., 1000 for mm)"
    )
    
    predict_parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save colored depth visualization"
    )
    
    predict_parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Save confidence map"
    )
    
    predict_parser.add_argument(
        "--colormap",
        default="turbo",
        help="Colormap for visualization"
    )
    
    # Batch predict command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Predict depth for batch of images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    batch_parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input images"
    )
    
    batch_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for depth maps"
    )
    
    batch_parser.add_argument(
        "--model",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS", "MiDaS_small"],
        default="DPT_Large",
        help="MiDaS model to use"
    )
    
    batch_parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on (auto, cpu, cuda, mps)"
    )
    
    batch_parser.add_argument(
        "--enable-amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision"
    )
    
    batch_parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Use multi-scale prediction"
    )
    
    batch_parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5],
        help="Scales for multi-scale prediction"
    )
    
    batch_parser.add_argument(
        "--fusion-method",
        choices=["average", "median", "max"],
        default="average",
        help="Fusion method for multi-scale prediction"
    )
    
    batch_parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for saving depth"
    )
    
    batch_parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save colored depth visualizations"
    )
    
    batch_parser.add_argument(
        "--colormap",
        default="turbo",
        help="Colormap for visualization"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark model inference speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    benchmark_parser.add_argument(
        "--input",
        required=True,
        help="Path to test image"
    )
    
    benchmark_parser.add_argument(
        "--model",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS", "MiDaS_small"],
        default="DPT_Large",
        help="MiDaS model to use"
    )
    
    benchmark_parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on"
    )
    
    benchmark_parser.add_argument(
        "--enable-amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision"
    )
    
    benchmark_parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of inference runs for timing"
    )
    
    benchmark_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger("depthmap", level=log_level)
    
    # Execute command
    if args.command == "predict":
        predict_command(args)
    elif args.command == "batch":
        batch_predict_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 