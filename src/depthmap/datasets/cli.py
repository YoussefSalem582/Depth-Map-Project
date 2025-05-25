"""Command line interface for dataset downloading and management."""

import argparse
import logging
import sys
from pathlib import Path

from .kitti import KITTIDownloader
from .nyu_depth import NYUDepthV2Downloader
from ..utils.logging import setup_logger

logger = logging.getLogger(__name__)


def download_command(args) -> None:
    """Download specified dataset."""
    output_dir = Path(args.output)
    
    if args.dataset == "kitti":
        logger.info(f"Downloading KITTI {args.split} dataset to {output_dir}")
        try:
            KITTIDownloader.download(
                output_dir=output_dir,
                split=args.split,
                subset=args.subset,
                force=args.force
            )
            logger.info("KITTI download completed successfully!")
        except Exception as e:
            logger.error(f"KITTI download failed: {e}")
            sys.exit(1)
    
    elif args.dataset == "nyu_depth_v2":
        logger.info(f"Downloading NYU Depth v2 {args.dataset_type} dataset to {output_dir}")
        try:
            NYUDepthV2Downloader.download(
                output_dir=output_dir,
                dataset_type=args.dataset_type,
                force=args.force
            )
            logger.info("NYU Depth v2 download completed successfully!")
        except Exception as e:
            logger.error(f"NYU Depth v2 download failed: {e}")
            sys.exit(1)
    
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


def verify_command(args) -> None:
    """Verify dataset integrity."""
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    if args.dataset == "kitti":
        logger.info(f"Verifying KITTI {args.split} dataset at {data_dir}")
        try:
            is_valid = KITTIDownloader.verify(data_dir, split=args.split)
            if is_valid:
                logger.info("KITTI dataset verification passed!")
            else:
                logger.error("KITTI dataset verification failed!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"KITTI verification failed: {e}")
            sys.exit(1)
    
    elif args.dataset == "nyu_depth_v2":
        logger.info(f"Verifying NYU Depth v2 {args.dataset_type} dataset at {data_dir}")
        try:
            is_valid = NYUDepthV2Downloader.verify(data_dir, dataset_type=args.dataset_type)
            if is_valid:
                logger.info("NYU Depth v2 dataset verification passed!")
            else:
                logger.error("NYU Depth v2 dataset verification failed!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"NYU Depth v2 verification failed: {e}")
            sys.exit(1)
    
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


def info_command(args) -> None:
    """Show information about available datasets."""
    print("Available Datasets:")
    print()
    
    print("1. KITTI Stereo Dataset")
    print("   - Splits: 2012, 2015")
    print("   - Subsets: training, testing")
    print("   - Type: Stereo vision (outdoor/automotive)")
    print("   - Resolution: 1242×375")
    print("   - Download size: ~12-15GB per split")
    print("   - Usage: depth-download --dataset kitti --split 2015 --output data/kitti")
    print()
    
    print("2. NYU Depth v2 Dataset")
    print("   - Types: labeled, raw")
    print("   - Type: Monocular (indoor)")
    print("   - Resolution: 640×480")
    print("   - Download size: ~2.8GB (labeled)")
    print("   - Usage: depth-download --dataset nyu_depth_v2 --output data/nyu")
    print()
    
    print("Dataset Structure:")
    print("data/")
    print("├── kitti/")
    print("│   ├── kitti_2015/")
    print("│   │   ├── training/")
    print("│   │   │   ├── image_2/     # Left camera images")
    print("│   │   │   ├── image_3/     # Right camera images")
    print("│   │   │   └── disp_occ_0/  # Ground truth disparity")
    print("│   │   └── testing/")
    print("│   └── kitti_2012/")
    print("└── nyu_depth_v2/")
    print("    ├── nyu_depth_v2_labeled.mat")
    print("    ├── rgb/                 # Extracted RGB images")
    print("    └── depth/               # Extracted depth maps")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dataset downloading and management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    download_parser.add_argument(
        "--dataset",
        choices=["kitti", "nyu_depth_v2"],
        required=True,
        help="Dataset to download"
    )
    
    download_parser.add_argument(
        "--output",
        required=True,
        help="Output directory for dataset"
    )
    
    download_parser.add_argument(
        "--split",
        choices=["2012", "2015"],
        default="2015",
        help="KITTI dataset split (only for KITTI)"
    )
    
    download_parser.add_argument(
        "--subset",
        choices=["training", "testing"],
        default="training",
        help="KITTI dataset subset (only for KITTI)"
    )
    
    download_parser.add_argument(
        "--dataset-type",
        choices=["labeled", "raw"],
        default="labeled",
        help="NYU Depth v2 dataset type (only for NYU)"
    )
    
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download if dataset exists"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify dataset integrity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    verify_parser.add_argument(
        "--dataset",
        choices=["kitti", "nyu_depth_v2"],
        required=True,
        help="Dataset to verify"
    )
    
    verify_parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing dataset"
    )
    
    verify_parser.add_argument(
        "--split",
        choices=["2012", "2015"],
        default="2015",
        help="KITTI dataset split (only for KITTI)"
    )
    
    verify_parser.add_argument(
        "--dataset-type",
        choices=["labeled", "raw"],
        default="labeled",
        help="NYU Depth v2 dataset type (only for NYU)"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about available datasets"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger("depthmap", level=log_level)
    
    # Execute command
    if args.command == "download":
        download_command(args)
    elif args.command == "verify":
        verify_command(args)
    elif args.command == "info":
        info_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 