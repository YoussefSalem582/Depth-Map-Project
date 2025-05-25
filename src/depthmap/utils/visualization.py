"""Visualization utilities for depth maps and results."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def colorize_depth(
    depth: np.ndarray,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    colormap: str = "turbo",
    invalid_value: float = 0.0
) -> np.ndarray:
    """Colorize a depth map for visualization.
    
    Args:
        depth: Depth map as numpy array.
        min_depth: Minimum depth value for normalization. If None, uses depth.min().
        max_depth: Maximum depth value for normalization. If None, uses depth.max().
        colormap: Matplotlib colormap name.
        invalid_value: Value representing invalid/missing depth.
        
    Returns:
        Colorized depth map as RGB numpy array (uint8).
    """
    # Handle invalid values
    valid_mask = depth != invalid_value
    if not np.any(valid_mask):
        # All values are invalid, return black image
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    # Normalize depth values
    if min_depth is None:
        min_depth = depth[valid_mask].min()
    if max_depth is None:
        max_depth = depth[valid_mask].max()
    
    # Avoid division by zero
    if max_depth == min_depth:
        normalized_depth = np.zeros_like(depth)
    else:
        normalized_depth = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Set invalid values to 0 (will be black in most colormaps)
    normalized_depth[~valid_mask] = 0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_depth = cmap(normalized_depth)
    
    # Convert to uint8 RGB
    colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)
    
    # Set invalid pixels to black
    colored_depth_rgb[~valid_mask] = [0, 0, 0]
    
    return colored_depth_rgb


def create_depth_overlay(
    image: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.6,
    colormap: str = "turbo",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None
) -> np.ndarray:
    """Create an overlay of depth map on the original image.
    
    Args:
        image: Original image as numpy array (RGB).
        depth: Depth map as numpy array.
        alpha: Blending factor (0.0 = only image, 1.0 = only depth).
        colormap: Matplotlib colormap name for depth visualization.
        min_depth: Minimum depth value for normalization.
        max_depth: Maximum depth value for normalization.
        
    Returns:
        Blended image as RGB numpy array (uint8).
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    # Resize depth to match image if necessary
    if depth.shape[:2] != image.shape[:2]:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Colorize depth
    colored_depth = colorize_depth(depth, min_depth, max_depth, colormap)
    
    # Blend images
    blended = cv2.addWeighted(image, 1 - alpha, colored_depth, alpha, 0)
    
    return blended


def create_side_by_side_comparison(
    images: list[np.ndarray],
    titles: list[str],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> Optional[plt.Figure]:
    """Create a side-by-side comparison of multiple images.
    
    Args:
        images: List of images as numpy arrays.
        titles: List of titles for each image.
        figsize: Figure size (width, height) in inches.
        save_path: Optional path to save the figure.
        dpi: DPI for saving the figure.
        
    Returns:
        Matplotlib figure if save_path is None, otherwise None.
    """
    n_images = len(images)
    if n_images != len(titles):
        raise ValueError("Number of images must match number of titles")
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        
        # Handle different image types
        if len(image.shape) == 2:
            # Grayscale or depth map
            im = ax.imshow(image, cmap='gray')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            # RGB image
            ax.imshow(image)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_depth_histogram(
    depth: np.ndarray,
    title: str = "Depth Distribution",
    bins: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """Plot histogram of depth values.
    
    Args:
        depth: Depth map as numpy array.
        title: Plot title.
        bins: Number of histogram bins.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height) in inches.
        
    Returns:
        Matplotlib figure if save_path is None, otherwise None.
    """
    # Remove invalid values (zeros and infinities)
    valid_depth = depth[(depth > 0) & np.isfinite(depth)]
    
    if len(valid_depth) == 0:
        print("Warning: No valid depth values found")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(valid_depth, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Depth (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_depth = np.mean(valid_depth)
    median_depth = np.median(valid_depth)
    std_depth = np.std(valid_depth)
    
    stats_text = f'Mean: {mean_depth:.2f}m\nMedian: {median_depth:.2f}m\nStd: {std_depth:.2f}m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def create_error_map(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    error_type: str = "absolute",
    colormap: str = "hot",
    max_error: Optional[float] = None
) -> np.ndarray:
    """Create an error map visualization.
    
    Args:
        pred_depth: Predicted depth map.
        gt_depth: Ground truth depth map.
        error_type: Type of error - 'absolute', 'relative', or 'squared'.
        colormap: Matplotlib colormap name.
        max_error: Maximum error value for normalization.
        
    Returns:
        Colorized error map as RGB numpy array (uint8).
    """
    # Ensure same shape
    if pred_depth.shape != gt_depth.shape:
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Calculate error
    if error_type == "absolute":
        error = np.abs(pred_depth - gt_depth)
    elif error_type == "relative":
        error = np.abs(pred_depth - gt_depth) / (gt_depth + 1e-8)
    elif error_type == "squared":
        error = (pred_depth - gt_depth) ** 2
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
    
    # Handle invalid values
    valid_mask = (gt_depth > 0) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    error[~valid_mask] = 0
    
    # Normalize error
    if max_error is None and np.any(valid_mask):
        max_error = np.percentile(error[valid_mask], 95)  # Use 95th percentile to avoid outliers
    
    if max_error is not None and max_error > 0:
        normalized_error = np.clip(error / max_error, 0, 1)
    else:
        normalized_error = np.zeros_like(error)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_error = cmap(normalized_error)
    
    # Convert to uint8 RGB
    colored_error_rgb = (colored_error[:, :, :3] * 255).astype(np.uint8)
    
    # Set invalid pixels to black
    colored_error_rgb[~valid_mask] = [0, 0, 0]
    
    return colored_error_rgb 