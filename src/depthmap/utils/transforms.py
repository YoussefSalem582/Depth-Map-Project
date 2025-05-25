"""Transformation utilities for depth maps and images."""

from typing import Optional, Tuple, Union

import cv2
import numpy as np


def normalize_depth(
    depth: np.ndarray,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    clip: bool = True
) -> np.ndarray:
    """Normalize depth map to [0, 1] range.
    
    Args:
        depth: Input depth map.
        min_depth: Minimum depth value. If None, uses depth.min().
        max_depth: Maximum depth value. If None, uses depth.max().
        clip: Whether to clip values outside [min_depth, max_depth].
        
    Returns:
        Normalized depth map in [0, 1] range.
    """
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()
    
    if clip:
        depth = np.clip(depth, min_depth, max_depth)
    
    # Avoid division by zero
    if max_depth == min_depth:
        return np.zeros_like(depth)
    
    normalized = (depth - min_depth) / (max_depth - min_depth)
    return normalized.astype(np.float32)


def denormalize_depth(
    normalized_depth: np.ndarray,
    min_depth: float,
    max_depth: float
) -> np.ndarray:
    """Denormalize depth map from [0, 1] range to original scale.
    
    Args:
        normalized_depth: Normalized depth map in [0, 1] range.
        min_depth: Minimum depth value for denormalization.
        max_depth: Maximum depth value for denormalization.
        
    Returns:
        Denormalized depth map.
    """
    depth = normalized_depth * (max_depth - min_depth) + min_depth
    return depth.astype(np.float32)


def resize_depth(
    depth: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """Resize depth map to target size.
    
    Args:
        depth: Input depth map.
        target_size: Target size as (width, height).
        interpolation: Interpolation method. INTER_NEAREST recommended for depth.
        
    Returns:
        Resized depth map.
    """
    width, height = target_size
    resized = cv2.resize(depth, (width, height), interpolation=interpolation)
    return resized.astype(np.float32)


def crop_depth(
    depth: np.ndarray,
    crop_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """Crop depth map using bounding box.
    
    Args:
        depth: Input depth map.
        crop_box: Crop box as (x, y, width, height).
        
    Returns:
        Cropped depth map.
    """
    x, y, w, h = crop_box
    return depth[y:y+h, x:x+w].copy()


def pad_depth(
    depth: np.ndarray,
    padding: Union[int, Tuple[int, int, int, int]],
    mode: str = "constant",
    constant_value: float = 0.0
) -> np.ndarray:
    """Pad depth map.
    
    Args:
        depth: Input depth map.
        padding: Padding size. If int, same padding on all sides.
                If tuple, (top, bottom, left, right).
        mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric').
        constant_value: Value for constant padding.
        
    Returns:
        Padded depth map.
    """
    if isinstance(padding, int):
        pad_width = ((padding, padding), (padding, padding))
    else:
        top, bottom, left, right = padding
        pad_width = ((top, bottom), (left, right))
    
    if mode == "constant":
        padded = np.pad(depth, pad_width, mode=mode, constant_values=constant_value)
    else:
        padded = np.pad(depth, pad_width, mode=mode)
    
    return padded.astype(np.float32)


def flip_depth(
    depth: np.ndarray,
    axis: int = 1
) -> np.ndarray:
    """Flip depth map along specified axis.
    
    Args:
        depth: Input depth map.
        axis: Axis to flip along (0=vertical, 1=horizontal).
        
    Returns:
        Flipped depth map.
    """
    return np.flip(depth, axis=axis).copy()


def rotate_depth(
    depth: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    fill_value: float = 0.0
) -> np.ndarray:
    """Rotate depth map by specified angle.
    
    Args:
        depth: Input depth map.
        angle: Rotation angle in degrees (positive = counter-clockwise).
        center: Rotation center as (x, y). If None, uses image center.
        scale: Scaling factor.
        fill_value: Value for pixels outside original image.
        
    Returns:
        Rotated depth map.
    """
    h, w = depth.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply rotation
    rotated = cv2.warpAffine(
        depth, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value
    )
    
    return rotated.astype(np.float32)


def apply_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0
) -> np.ndarray:
    """Apply binary mask to depth map.
    
    Args:
        depth: Input depth map.
        mask: Binary mask (0 = invalid, >0 = valid).
        fill_value: Value for masked pixels.
        
    Returns:
        Masked depth map.
    """
    masked_depth = depth.copy()
    masked_depth[mask == 0] = fill_value
    return masked_depth


def filter_depth(
    depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
    fill_value: float = 0.0
) -> np.ndarray:
    """Filter depth map by depth range.
    
    Args:
        depth: Input depth map.
        min_depth: Minimum valid depth.
        max_depth: Maximum valid depth.
        fill_value: Value for invalid pixels.
        
    Returns:
        Filtered depth map.
    """
    filtered_depth = depth.copy()
    invalid_mask = (depth < min_depth) | (depth > max_depth)
    filtered_depth[invalid_mask] = fill_value
    return filtered_depth


def smooth_depth(
    depth: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """Smooth depth map using Gaussian filter.
    
    Args:
        depth: Input depth map.
        kernel_size: Size of Gaussian kernel (must be odd).
        sigma: Standard deviation for Gaussian kernel.
        
    Returns:
        Smoothed depth map.
    """
    # Create mask for valid pixels
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        return depth
    
    # Apply Gaussian blur only to valid pixels
    smoothed = cv2.GaussianBlur(depth, (kernel_size, kernel_size), sigma)
    
    # Preserve invalid pixels
    result = depth.copy()
    result[valid_mask] = smoothed[valid_mask]
    
    return result.astype(np.float32)


def median_filter_depth(
    depth: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """Apply median filter to depth map.
    
    Args:
        depth: Input depth map.
        kernel_size: Size of median filter kernel.
        
    Returns:
        Filtered depth map.
    """
    # Create mask for valid pixels
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        return depth
    
    # Apply median filter
    filtered = cv2.medianBlur(depth.astype(np.float32), kernel_size)
    
    # Preserve invalid pixels
    result = depth.copy()
    result[valid_mask] = filtered[valid_mask]
    
    return result.astype(np.float32) 