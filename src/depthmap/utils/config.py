"""Configuration management utilities."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration as OmegaConf DictConfig.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return OmegaConf.create(config_dict)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}") from e


def save_config(config: Union[DictConfig, Dict[str, Any]], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        config_path: Path where to save the configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_yaml(config)
    else:
        config_dict = yaml.dump(config, default_flow_style=False)
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_dict)


def merge_configs(*configs: Union[DictConfig, Dict[str, Any]]) -> DictConfig:
    """Merge multiple configurations with later configs overriding earlier ones.
    
    Args:
        *configs: Variable number of configurations to merge.
        
    Returns:
        Merged configuration.
    """
    merged = OmegaConf.create({})
    for config in configs:
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        merged = OmegaConf.merge(merged, config)
    return merged


def get_default_config() -> DictConfig:
    """Get default configuration for the depth estimation project.
    
    Returns:
        Default configuration.
    """
    default_config = {
        "data": {
            "root_dir": "data",
            "datasets": ["kitti", "nyu_depth_v2"],
            "image_size": [480, 640],
            "batch_size": 8,
            "num_workers": 4,
        },
        "classical": {
            "stereo_method": "SGBM",
            "num_disparities": 96,
            "block_size": 11,
            "min_disparity": 0,
            "uniqueness_ratio": 10,
            "speckle_window_size": 100,
            "speckle_range": 32,
        },
        "generative": {
            "model_name": "DPT_Large",
            "device": "auto",
            "batch_size": 1,
            "enable_amp": True,
        },
        "evaluation": {
            "metrics": ["rmse", "mae", "silog", "delta1", "delta2", "delta3"],
            "depth_cap": 80.0,
            "min_depth": 0.1,
        },
        "visualization": {
            "colormap": "turbo",
            "save_plots": True,
            "plot_format": "png",
            "dpi": 300,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }
    return OmegaConf.create(default_config) 