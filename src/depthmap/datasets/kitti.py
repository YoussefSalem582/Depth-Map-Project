"""KITTI dataset loader and downloader for stereo depth estimation."""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import cv2
import numpy as np
import requests
from tqdm import tqdm

from ..utils.io import load_image, load_depth

logger = logging.getLogger(__name__)


class KITTIDataset:
    """KITTI stereo dataset loader for depth estimation.
    
    Supports KITTI 2012 and 2015 stereo datasets with ground truth disparity maps.
    """
    
    URLS = {
        "2012": {
            "training": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip",
            "testing": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow_testing.zip",
        },
        "2015": {
            "training": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip",
            "testing": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_testing.zip",
        }
    }
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "2015",
        subset: str = "training",
        transform: Optional[callable] = None
    ):
        """Initialize KITTI dataset.
        
        Args:
            root_dir: Root directory containing KITTI data
            split: Dataset split ("2012" or "2015")
            subset: Data subset ("training" or "testing")
            transform: Optional transform function
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.subset = subset
        self.transform = transform
        
        # Dataset paths
        self.data_dir = self.root_dir / f"kitti_{split}" / subset
        self.left_dir = self.data_dir / "image_2"
        self.right_dir = self.data_dir / "image_3"
        self.disp_dir = self.data_dir / "disp_occ_0" if subset == "training" else None
        
        # Load file lists
        self._load_file_lists()
        
        # Camera parameters (KITTI stereo setup)
        self.baseline = 0.54  # meters
        self.focal_length = 721.5377  # pixels (approximate)
        
    def _load_file_lists(self) -> None:
        """Load lists of available files."""
        if not self.left_dir.exists():
            raise FileNotFoundError(
                f"KITTI dataset not found at {self.data_dir}. "
                f"Please download using KITTIDownloader.download()"
            )
            
        # Get all left images
        left_files = sorted(list(self.left_dir.glob("*.png")))
        self.file_names = [f.stem for f in left_files]
        
        logger.info(f"Found {len(self.file_names)} image pairs in {self.data_dir}")
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.file_names)
        
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - left: Left camera image (H, W, 3)
                - right: Right camera image (H, W, 3)
                - disparity: Ground truth disparity (H, W) [if training]
                - depth: Ground truth depth (H, W) [if training]
                - filename: Base filename
        """
        filename = self.file_names[idx]
        
        # Load stereo pair
        left_path = self.left_dir / f"{filename}.png"
        right_path = self.right_dir / f"{filename}.png"
        
        left_img = load_image(left_path)
        right_img = load_image(right_path)
        
        sample = {
            "left": left_img,
            "right": right_img,
            "filename": filename,
            "baseline": self.baseline,
            "focal_length": self.focal_length,
        }
        
        # Load ground truth if available
        if self.disp_dir and self.subset == "training":
            disp_path = self.disp_dir / f"{filename}.png"
            if disp_path.exists():
                disparity = load_depth(disp_path, scale_factor=1.0)
                
                # Convert disparity to depth
                # depth = (baseline * focal_length) / disparity
                # Handle invalid disparities (0 values)
                depth = np.zeros_like(disparity)
                valid_mask = disparity > 0
                depth[valid_mask] = (self.baseline * self.focal_length) / disparity[valid_mask]
                
                sample["disparity"] = disparity
                sample["depth"] = depth
                sample["valid_mask"] = valid_mask
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def get_calibration(self) -> Dict[str, np.ndarray]:
        """Get camera calibration parameters.
        
        Returns:
            Dictionary with calibration matrices
        """
        # KITTI camera calibration (approximate)
        K_left = np.array([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.854],
            [0.0, 0.0, 1.0]
        ])
        
        K_right = np.array([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.854],
            [0.0, 0.0, 1.0]
        ])
        
        # Rectification matrices (identity for rectified images)
        R_left = np.eye(3)
        R_right = np.eye(3)
        
        # Projection matrices
        P_left = np.array([
            [721.5377, 0.0, 609.5593, 0.0],
            [0.0, 721.5377, 172.854, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        P_right = np.array([
            [721.5377, 0.0, 609.5593, -387.5744],  # -fx * baseline
            [0.0, 721.5377, 172.854, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        return {
            "camera_matrix_left": K_left,
            "camera_matrix_right": K_right,
            "rectification_left": R_left,
            "rectification_right": R_right,
            "projection_left": P_left,
            "projection_right": P_right,
            "baseline": self.baseline,
            "focal_length": self.focal_length,
        }


class KITTIDownloader:
    """Download and extract KITTI dataset."""
    
    @staticmethod
    def download(
        output_dir: Union[str, Path],
        split: str = "2015",
        subset: str = "training",
        force: bool = False
    ) -> None:
        """Download KITTI dataset.
        
        Args:
            output_dir: Output directory
            split: Dataset split ("2012" or "2015")
            subset: Data subset ("training" or "testing")
            force: Force re-download if exists
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if split not in KITTIDataset.URLS:
            raise ValueError(f"Invalid split: {split}. Choose from {list(KITTIDataset.URLS.keys())}")
            
        if subset not in KITTIDataset.URLS[split]:
            raise ValueError(f"Invalid subset: {subset}. Choose from {list(KITTIDataset.URLS[split].keys())}")
            
        url = KITTIDataset.URLS[split][subset]
        filename = url.split("/")[-1]
        zip_path = output_dir / filename
        extract_dir = output_dir / f"kitti_{split}"
        
        # Check if already downloaded
        if extract_dir.exists() and not force:
            logger.info(f"KITTI {split} {subset} already exists at {extract_dir}")
            return
            
        # Download
        logger.info(f"Downloading KITTI {split} {subset} from {url}")
        KITTIDownloader._download_file(url, zip_path)
        
        # Extract
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # Clean up
        zip_path.unlink()
        logger.info(f"Successfully downloaded KITTI {split} {subset} to {extract_dir}")
        
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
    def verify(data_dir: Union[str, Path], split: str = "2015") -> bool:
        """Verify dataset integrity.
        
        Args:
            data_dir: Dataset directory
            split: Dataset split
            
        Returns:
            True if dataset is valid
        """
        data_dir = Path(data_dir)
        
        # Check required directories
        required_dirs = [
            data_dir / f"kitti_{split}" / "training" / "image_2",
            data_dir / f"kitti_{split}" / "training" / "image_3",
            data_dir / f"kitti_{split}" / "training" / "disp_occ_0",
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Missing directory: {dir_path}")
                return False
                
        # Check file counts
        left_files = list((data_dir / f"kitti_{split}" / "training" / "image_2").glob("*.png"))
        right_files = list((data_dir / f"kitti_{split}" / "training" / "image_3").glob("*.png"))
        disp_files = list((data_dir / f"kitti_{split}" / "training" / "disp_occ_0").glob("*.png"))
        
        if len(left_files) != len(right_files) or len(left_files) != len(disp_files):
            logger.error(f"File count mismatch: {len(left_files)} left, {len(right_files)} right, {len(disp_files)} disparity")
            return False
            
        logger.info(f"KITTI {split} dataset verified: {len(left_files)} samples")
        return True 