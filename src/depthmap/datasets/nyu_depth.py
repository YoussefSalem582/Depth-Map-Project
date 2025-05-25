"""NYU Depth v2 dataset loader and downloader for monocular depth estimation."""

import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import cv2
import numpy as np
import requests
from tqdm import tqdm
import h5py
from scipy.io import loadmat

from ..utils.io import load_image, save_depth

logger = logging.getLogger(__name__)


class NYUDepthV2Dataset:
    """NYU Depth v2 dataset loader for monocular depth estimation.
    
    Supports both the labeled dataset (1449 images) and the raw dataset.
    """
    
    URLS = {
        "labeled": "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
        "raw": "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip",
    }
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        dataset_type: str = "labeled",
        transform: Optional[callable] = None
    ):
        """Initialize NYU Depth v2 dataset.
        
        Args:
            root_dir: Root directory containing NYU data
            split: Data split ("train", "test", or "all")
            dataset_type: Dataset type ("labeled" or "raw")
            transform: Optional transform function
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_type = dataset_type
        self.transform = transform
        
        # Dataset paths
        self.data_dir = self.root_dir / "nyu_depth_v2"
        
        if dataset_type == "labeled":
            self.mat_file = self.data_dir / "nyu_depth_v2_labeled.mat"
            self._load_labeled_data()
        else:
            self.rgb_dir = self.data_dir / "rgb"
            self.depth_dir = self.data_dir / "depth"
            self._load_raw_data()
            
        # Camera parameters (NYU Depth v2)
        self.focal_length = 518.8579  # pixels
        self.cx = 325.5824  # principal point x
        self.cy = 253.7362  # principal point y
        
    def _load_labeled_data(self) -> None:
        """Load labeled dataset from .mat file."""
        if not self.mat_file.exists():
            raise FileNotFoundError(
                f"NYU Depth v2 labeled dataset not found at {self.mat_file}. "
                f"Please download using NYUDepthV2Downloader.download()"
            )
            
        # Load .mat file
        logger.info(f"Loading NYU Depth v2 labeled dataset from {self.mat_file}")
        data = loadmat(str(self.mat_file))
        
        self.images = data['images']  # (480, 640, 3, 1449)
        self.depths = data['depths']  # (480, 640, 1449)
        
        # Split data
        total_samples = self.images.shape[3]
        if self.split == "train":
            # Use first 1000 for training
            self.indices = list(range(1000))
        elif self.split == "test":
            # Use last 449 for testing
            self.indices = list(range(1000, total_samples))
        else:  # "all"
            self.indices = list(range(total_samples))
            
        logger.info(f"Loaded {len(self.indices)} samples for split '{self.split}'")
        
    def _load_raw_data(self) -> None:
        """Load raw dataset from extracted files."""
        if not self.rgb_dir.exists() or not self.depth_dir.exists():
            raise FileNotFoundError(
                f"NYU Depth v2 raw dataset not found at {self.data_dir}. "
                f"Please download using NYUDepthV2Downloader.download()"
            )
            
        # Get all RGB files
        rgb_files = sorted(list(self.rgb_dir.glob("*.png")))
        self.file_names = [f.stem for f in rgb_files]
        
        logger.info(f"Found {len(self.file_names)} samples in raw dataset")
        
    def __len__(self) -> int:
        """Return number of samples."""
        if self.dataset_type == "labeled":
            return len(self.indices)
        else:
            return len(self.file_names)
            
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - rgb: RGB image (H, W, 3)
                - depth: Ground truth depth (H, W)
                - filename: Sample identifier
        """
        if self.dataset_type == "labeled":
            return self._get_labeled_sample(idx)
        else:
            return self._get_raw_sample(idx)
            
    def _get_labeled_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Get sample from labeled dataset."""
        sample_idx = self.indices[idx]
        
        # Extract image and depth
        rgb = self.images[:, :, :, sample_idx]  # (480, 640, 3)
        depth = self.depths[:, :, sample_idx]   # (480, 640)
        
        # Convert to standard format
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        
        # Create valid mask (NYU uses 0 for invalid depth)
        valid_mask = depth > 0
        
        sample = {
            "rgb": rgb,
            "depth": depth,
            "valid_mask": valid_mask,
            "filename": f"labeled_{sample_idx:04d}",
            "focal_length": self.focal_length,
            "cx": self.cx,
            "cy": self.cy,
        }
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def _get_raw_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Get sample from raw dataset."""
        filename = self.file_names[idx]
        
        # Load RGB and depth
        rgb_path = self.rgb_dir / f"{filename}.png"
        depth_path = self.depth_dir / f"{filename}.png"
        
        rgb = load_image(rgb_path)
        depth = load_image(depth_path, grayscale=True).astype(np.float32)
        
        # Convert depth from millimeters to meters
        depth = depth / 1000.0
        
        # Create valid mask
        valid_mask = depth > 0
        
        sample = {
            "rgb": rgb,
            "depth": depth,
            "valid_mask": valid_mask,
            "filename": filename,
            "focal_length": self.focal_length,
            "cx": self.cx,
            "cy": self.cy,
        }
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def get_calibration(self) -> Dict[str, np.ndarray]:
        """Get camera calibration parameters.
        
        Returns:
            Dictionary with calibration matrices
        """
        # NYU Depth v2 camera calibration
        K = np.array([
            [self.focal_length, 0.0, self.cx],
            [0.0, self.focal_length, self.cy],
            [0.0, 0.0, 1.0]
        ])
        
        return {
            "camera_matrix": K,
            "focal_length": self.focal_length,
            "cx": self.cx,
            "cy": self.cy,
        }


class NYUDepthV2Downloader:
    """Download and extract NYU Depth v2 dataset."""
    
    @staticmethod
    def download(
        output_dir: Union[str, Path],
        dataset_type: str = "labeled",
        force: bool = False
    ) -> None:
        """Download NYU Depth v2 dataset.
        
        Args:
            output_dir: Output directory
            dataset_type: Dataset type ("labeled" or "raw")
            force: Force re-download if exists
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset_type not in NYUDepthV2Dataset.URLS:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Choose from {list(NYUDepthV2Dataset.URLS.keys())}")
            
        url = NYUDepthV2Dataset.URLS[dataset_type]
        data_dir = output_dir / "nyu_depth_v2"
        data_dir.mkdir(exist_ok=True)
        
        if dataset_type == "labeled":
            NYUDepthV2Downloader._download_labeled(url, data_dir, force)
        else:
            NYUDepthV2Downloader._download_raw(url, data_dir, force)
            
    @staticmethod
    def _download_labeled(url: str, data_dir: Path, force: bool) -> None:
        """Download labeled dataset."""
        mat_file = data_dir / "nyu_depth_v2_labeled.mat"
        
        if mat_file.exists() and not force:
            logger.info(f"NYU Depth v2 labeled dataset already exists at {mat_file}")
            return
            
        logger.info(f"Downloading NYU Depth v2 labeled dataset from {url}")
        NYUDepthV2Downloader._download_file(url, mat_file)
        
        # Extract images and depths to separate files for easier access
        logger.info("Extracting images and depths from .mat file")
        NYUDepthV2Downloader._extract_labeled_data(mat_file, data_dir)
        
    @staticmethod
    def _download_raw(url: str, data_dir: Path, force: bool) -> None:
        """Download raw dataset."""
        zip_file = data_dir / "nyu_depth_v2_raw.zip"
        
        if (data_dir / "rgb").exists() and not force:
            logger.info(f"NYU Depth v2 raw dataset already exists at {data_dir}")
            return
            
        logger.info(f"Downloading NYU Depth v2 raw dataset from {url}")
        NYUDepthV2Downloader._download_file(url, zip_file)
        
        # Extract
        logger.info(f"Extracting {zip_file}")
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Clean up
        zip_file.unlink()
        
    @staticmethod
    def _extract_labeled_data(mat_file: Path, output_dir: Path) -> None:
        """Extract labeled data to individual files."""
        # Create directories
        rgb_dir = output_dir / "rgb"
        depth_dir = output_dir / "depth"
        rgb_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        
        # Load .mat file
        data = loadmat(str(mat_file))
        images = data['images']  # (480, 640, 3, 1449)
        depths = data['depths']  # (480, 640, 1449)
        
        # Extract each sample
        num_samples = images.shape[3]
        for i in tqdm(range(num_samples), desc="Extracting samples"):
            # Extract image and depth
            rgb = images[:, :, :, i].astype(np.uint8)
            depth = depths[:, :, i].astype(np.float32)
            
            # Save files
            rgb_path = rgb_dir / f"{i:04d}.png"
            depth_path = depth_dir / f"{i:04d}.png"
            
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            save_depth(depth_path, depth)
            
        logger.info(f"Extracted {num_samples} samples to {output_dir}")
        
    @staticmethod
    def _download_file(url: str, output_path: Path) -> None:
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    @staticmethod
    def verify(data_dir: Union[str, Path], dataset_type: str = "labeled") -> bool:
        """Verify dataset integrity.
        
        Args:
            data_dir: Dataset directory
            dataset_type: Dataset type
            
        Returns:
            True if dataset is valid
        """
        data_dir = Path(data_dir) / "nyu_depth_v2"
        
        if dataset_type == "labeled":
            mat_file = data_dir / "nyu_depth_v2_labeled.mat"
            if not mat_file.exists():
                logger.error(f"Missing file: {mat_file}")
                return False
                
            # Check extracted files
            rgb_dir = data_dir / "rgb"
            depth_dir = data_dir / "depth"
            
            if rgb_dir.exists() and depth_dir.exists():
                rgb_files = list(rgb_dir.glob("*.png"))
                depth_files = list(depth_dir.glob("*.png"))
                
                if len(rgb_files) != len(depth_files):
                    logger.error(f"File count mismatch: {len(rgb_files)} RGB, {len(depth_files)} depth")
                    return False
                    
                logger.info(f"NYU Depth v2 {dataset_type} dataset verified: {len(rgb_files)} samples")
            else:
                logger.info(f"NYU Depth v2 {dataset_type} dataset verified: .mat file exists")
                
        else:  # raw
            rgb_dir = data_dir / "rgb"
            depth_dir = data_dir / "depth"
            
            if not rgb_dir.exists() or not depth_dir.exists():
                logger.error(f"Missing directories: {rgb_dir}, {depth_dir}")
                return False
                
            rgb_files = list(rgb_dir.glob("*.png"))
            depth_files = list(depth_dir.glob("*.png"))
            
            if len(rgb_files) != len(depth_files):
                logger.error(f"File count mismatch: {len(rgb_files)} RGB, {len(depth_files)} depth")
                return False
                
            logger.info(f"NYU Depth v2 {dataset_type} dataset verified: {len(rgb_files)} samples")
            
        return True 