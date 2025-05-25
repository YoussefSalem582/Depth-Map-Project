"""Dataset management for depth estimation."""

from .kitti import KITTIDataset, KITTIDownloader
from .nyu_depth import NYUDepthV2Dataset, NYUDepthV2Downloader

# Dataset registry
DATASETS = {
    "kitti": {
        "description": "KITTI stereo dataset for outdoor depth estimation",
        "splits": ["2012", "2015"],
        "subsets": ["training", "testing"],
        "type": "stereo",
        "loader": KITTIDataset,
        "downloader": KITTIDownloader,
    },
    "nyu_depth_v2": {
        "description": "NYU Depth v2 dataset for indoor monocular depth estimation",
        "splits": ["train", "test", "all"],
        "subsets": ["labeled", "raw"],
        "type": "monocular",
        "loader": NYUDepthV2Dataset,
        "downloader": NYUDepthV2Downloader,
    },
}

__all__ = [
    "DATASETS",
    "KITTIDataset",
    "KITTIDownloader",
    "NYUDepthV2Dataset", 
    "NYUDepthV2Downloader",
] 