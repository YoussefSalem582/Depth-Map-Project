"""Input/Output utilities for images and depth maps."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(
    image_path: Union[str, Path], 
    target_size: Optional[Tuple[int, int]] = None,
    color_mode: str = "RGB"
) -> np.ndarray:
    """Load an image from file.
    
    Args:
        image_path: Path to the image file.
        target_size: Optional target size (width, height) for resizing.
        color_mode: Color mode - 'RGB', 'BGR', or 'GRAY'.
        
    Returns:
        Image as numpy array.
        
    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If color_mode is invalid.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if color_mode not in ["RGB", "BGR", "GRAY"]:
        raise ValueError(f"Invalid color_mode: {color_mode}. Must be 'RGB', 'BGR', or 'GRAY'")
    
    # Load image using OpenCV for better performance
    if color_mode == "GRAY":
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if color_mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize if target size is specified
    if target_size is not None:
        width, height = target_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return image


def save_image(
    image: np.ndarray, 
    output_path: Union[str, Path],
    color_mode: str = "RGB",
    quality: int = 95
) -> None:
    """Save an image to file.
    
    Args:
        image: Image as numpy array.
        output_path: Path where to save the image.
        color_mode: Color mode of input image - 'RGB', 'BGR', or 'GRAY'.
        quality: JPEG quality (0-100) if saving as JPEG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert color mode if necessary
    if color_mode == "RGB" and len(image.shape) == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image.copy()
    
    # Set quality for JPEG files
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), image_to_save, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), image_to_save)


def load_depth(
    depth_path: Union[str, Path],
    scale_factor: float = 1.0,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Load a depth map from file.
    
    Args:
        depth_path: Path to the depth file.
        scale_factor: Scale factor to apply to depth values.
        target_size: Optional target size (width, height) for resizing.
        
    Returns:
        Depth map as numpy array (float32).
        
    Raises:
        FileNotFoundError: If depth file doesn't exist.
    """
    depth_path = Path(depth_path)
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    
    # Handle different depth file formats
    if depth_path.suffix.lower() in ['.png', '.tiff', '.tif']:
        # Load as 16-bit for depth maps
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to load depth map: {depth_path}")
        depth = depth.astype(np.float32)
    elif depth_path.suffix.lower() in ['.npy']:
        depth = np.load(str(depth_path)).astype(np.float32)
    elif depth_path.suffix.lower() in ['.pfm']:
        depth = load_pfm(depth_path)
    else:
        # Try loading as regular image
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Failed to load depth map: {depth_path}")
        depth = depth.astype(np.float32)
    
    # Apply scale factor
    if scale_factor != 1.0:
        depth = depth * scale_factor
    
    # Resize if target size is specified
    if target_size is not None:
        width, height = target_size
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return depth


def save_depth(
    depth: np.ndarray,
    output_path: Union[str, Path],
    scale_factor: float = 1.0,
    format_type: str = "png"
) -> None:
    """Save a depth map to file.
    
    Args:
        depth: Depth map as numpy array.
        output_path: Path where to save the depth map.
        scale_factor: Scale factor to apply before saving.
        format_type: Output format - 'png', 'npy', or 'pfm'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply scale factor
    depth_scaled = depth * scale_factor
    
    if format_type == "png":
        # Save as 16-bit PNG
        depth_uint16 = np.clip(depth_scaled, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_uint16)
    elif format_type == "npy":
        np.save(str(output_path), depth_scaled.astype(np.float32))
    elif format_type == "pfm":
        save_pfm(depth_scaled, output_path)
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")


def load_pfm(file_path: Union[str, Path]) -> np.ndarray:
    """Load a PFM (Portable Float Map) file.
    
    Args:
        file_path: Path to the PFM file.
        
    Returns:
        Image data as numpy array.
    """
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError('Not a PFM file.')
        
        dim_match = f.readline().decode('utf-8').rstrip().split()
        width, height = int(dim_match[0]), int(dim_match[1])
        
        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'  # little endian
            scale = -scale
        else:
            endian = '>'  # big endian
        
        data = np.frombuffer(f.read(), endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        
    return data


def save_pfm(data: np.ndarray, file_path: Union[str, Path]) -> None:
    """Save a numpy array as PFM file.
    
    Args:
        data: Image data as numpy array.
        file_path: Path where to save the PFM file.
    """
    with open(file_path, 'wb') as f:
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if len(data.shape) == 3:
            height, width, channels = data.shape
            if channels != 3:
                raise ValueError('PFM color images must have 3 channels')
            f.write(b'PF\n')
        else:
            height, width = data.shape
            f.write(b'Pf\n')
        
        f.write(f'{width} {height}\n'.encode('utf-8'))
        f.write(b'-1.0\n')  # little endian
        
        data = np.flipud(data)
        data.tofile(f) 