"""Command line interface for depth estimation evaluation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics, compare_methods, print_metrics_table
from ..utils.io import load_depth
from ..utils.logging import setup_logger

logger = logging.getLogger(__name__)


def evaluate_command(args) -> None:
    """Evaluate depth predictions against ground truth."""
    pred_files = sorted(Path(args.predictions).glob("*.png"))
    gt_files = sorted(Path(args.ground_truth).glob("*.png"))
    
    if len(pred_files) == 0:
        logger.error(f"No prediction files found in {args.predictions}")
        sys.exit(1)
    
    if len(gt_files) == 0:
        logger.error(f"No ground truth files found in {args.ground_truth}")
        sys.exit(1)
    
    logger.info(f"Found {len(pred_files)} prediction files and {len(gt_files)} ground truth files")
    
    # Match files by name
    matched_pairs = []
    for pred_file in pred_files:
        # Try to find matching ground truth file
        gt_file = None
        for gt_candidate in gt_files:
            if pred_file.stem == gt_candidate.stem or pred_file.stem.replace("_depth", "") == gt_candidate.stem:
                gt_file = gt_candidate
                break
        
        if gt_file is not None:
            matched_pairs.append((pred_file, gt_file))
        else:
            logger.warning(f"No matching ground truth found for {pred_file}")
    
    if len(matched_pairs) == 0:
        logger.error("No matching prediction-ground truth pairs found")
        sys.exit(1)
    
    logger.info(f"Found {len(matched_pairs)} matching pairs")
    
    # Evaluate each pair
    all_metrics = []
    for i, (pred_file, gt_file) in enumerate(matched_pairs):
        logger.info(f"Evaluating {i+1}/{len(matched_pairs)}: {pred_file.name}")
        
        try:
            # Load depth maps
            pred_depth = load_depth(pred_file, scale_factor=args.pred_scale)
            gt_depth = load_depth(gt_file, scale_factor=args.gt_scale)
            
            # Resize if needed
            if pred_depth.shape != gt_depth.shape:
                logger.warning(f"Shape mismatch: pred {pred_depth.shape}, gt {gt_depth.shape}")
                if args.resize_pred:
                    import cv2
                    pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
                else:
                    logger.error("Shape mismatch and resize not enabled")
                    continue
            
            # Compute metrics
            metrics = compute_all_metrics(
                pred_depth, gt_depth,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                depth_cap=args.depth_cap
            )
            
            metrics["filename"] = pred_file.stem
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {pred_file}: {e}")
            continue
    
    if len(all_metrics) == 0:
        logger.error("No successful evaluations")
        sys.exit(1)
    
    # Compute summary statistics
    df = pd.DataFrame(all_metrics)
    summary = df.describe()
    
    logger.info("Evaluation Results:")
    logger.info(f"Successfully evaluated {len(all_metrics)} pairs")
    
    # Print summary
    print("\nSummary Statistics:")
    print(summary.round(4))
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df.to_csv(output_path, index=False)
        logger.info(f"Detailed results saved to {output_path}")
        
        # Save summary
        summary_path = output_path.with_suffix(".summary.csv")
        summary.to_csv(summary_path)
        logger.info(f"Summary statistics saved to {summary_path}")


def compare_command(args) -> None:
    """Compare multiple depth estimation methods."""
    method_dirs = {}
    
    # Parse method directories
    for method_spec in args.methods:
        if ":" not in method_spec:
            logger.error(f"Invalid method specification: {method_spec}. Use format 'name:path'")
            sys.exit(1)
        
        name, path = method_spec.split(":", 1)
        method_dirs[name] = Path(path)
        
        if not method_dirs[name].exists():
            logger.error(f"Method directory not found: {method_dirs[name]}")
            sys.exit(1)
    
    # Load ground truth files
    gt_files = sorted(Path(args.ground_truth).glob("*.png"))
    if len(gt_files) == 0:
        logger.error(f"No ground truth files found in {args.ground_truth}")
        sys.exit(1)
    
    logger.info(f"Comparing {len(method_dirs)} methods on {len(gt_files)} images")
    
    # Evaluate each method
    method_results = {}
    for method_name, method_dir in method_dirs.items():
        logger.info(f"Evaluating method: {method_name}")
        
        pred_files = sorted(method_dir.glob("*.png"))
        method_metrics = []
        
        for gt_file in gt_files:
            # Find matching prediction file
            pred_file = None
            for pred_candidate in pred_files:
                if (pred_candidate.stem == gt_file.stem or 
                    pred_candidate.stem.replace("_depth", "") == gt_file.stem):
                    pred_file = pred_candidate
                    break
            
            if pred_file is None:
                logger.warning(f"No prediction found for {gt_file.name} in {method_name}")
                continue
            
            try:
                # Load depth maps
                pred_depth = load_depth(pred_file, scale_factor=args.pred_scale)
                gt_depth = load_depth(gt_file, scale_factor=args.gt_scale)
                
                # Resize if needed
                if pred_depth.shape != gt_depth.shape:
                    if args.resize_pred:
                        import cv2
                        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), 
                                              interpolation=cv2.INTER_NEAREST)
                    else:
                        logger.warning(f"Shape mismatch for {gt_file.name}")
                        continue
                
                # Compute metrics
                metrics = compute_all_metrics(
                    pred_depth, gt_depth,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    depth_cap=args.depth_cap
                )
                
                method_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {pred_file}: {e}")
                continue
        
        if len(method_metrics) > 0:
            # Compute average metrics for this method
            avg_metrics = {}
            for key in method_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in method_metrics])
            
            method_results[method_name] = avg_metrics
            logger.info(f"Evaluated {len(method_metrics)} images for {method_name}")
        else:
            logger.warning(f"No successful evaluations for {method_name}")
    
    if len(method_results) == 0:
        logger.error("No successful method evaluations")
        sys.exit(1)
    
    # Print comparison table
    print("\nMethod Comparison:")
    print_metrics_table(method_results)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save comparison results
        df = pd.DataFrame(method_results).T
        df.to_csv(output_path)
        logger.info(f"Comparison results saved to {output_path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Depth estimation evaluation and comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate depth predictions against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    eval_parser.add_argument(
        "--predictions",
        required=True,
        help="Directory containing prediction depth maps"
    )
    
    eval_parser.add_argument(
        "--ground-truth",
        required=True,
        help="Directory containing ground truth depth maps"
    )
    
    eval_parser.add_argument(
        "--output", "-o",
        help="Output CSV file for results"
    )
    
    eval_parser.add_argument(
        "--pred-scale",
        type=float,
        default=1.0,
        help="Scale factor for prediction depth maps"
    )
    
    eval_parser.add_argument(
        "--gt-scale",
        type=float,
        default=1.0,
        help="Scale factor for ground truth depth maps"
    )
    
    eval_parser.add_argument(
        "--min-depth",
        type=float,
        default=0.1,
        help="Minimum depth for evaluation"
    )
    
    eval_parser.add_argument(
        "--max-depth",
        type=float,
        default=100.0,
        help="Maximum depth for evaluation"
    )
    
    eval_parser.add_argument(
        "--depth-cap",
        type=float,
        default=80.0,
        help="Depth cap for certain metrics"
    )
    
    eval_parser.add_argument(
        "--resize-pred",
        action="store_true",
        help="Resize predictions to match ground truth"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple depth estimation methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    compare_parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Method specifications in format 'name:path'"
    )
    
    compare_parser.add_argument(
        "--ground-truth",
        required=True,
        help="Directory containing ground truth depth maps"
    )
    
    compare_parser.add_argument(
        "--output", "-o",
        help="Output CSV file for comparison results"
    )
    
    compare_parser.add_argument(
        "--pred-scale",
        type=float,
        default=1.0,
        help="Scale factor for prediction depth maps"
    )
    
    compare_parser.add_argument(
        "--gt-scale",
        type=float,
        default=1.0,
        help="Scale factor for ground truth depth maps"
    )
    
    compare_parser.add_argument(
        "--min-depth",
        type=float,
        default=0.1,
        help="Minimum depth for evaluation"
    )
    
    compare_parser.add_argument(
        "--max-depth",
        type=float,
        default=100.0,
        help="Maximum depth for evaluation"
    )
    
    compare_parser.add_argument(
        "--depth-cap",
        type=float,
        default=80.0,
        help="Depth cap for certain metrics"
    )
    
    compare_parser.add_argument(
        "--resize-pred",
        action="store_true",
        help="Resize predictions to match ground truth"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger("depthmap", level=log_level)
    
    # Execute command
    if args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "compare":
        compare_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 