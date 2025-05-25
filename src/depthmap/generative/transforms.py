"""Transforms for generative depth estimation models."""

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


class DepthTransforms:
    """Transform utilities for depth estimation models."""
    
    def __init__(self):
        """Initialize depth transforms."""
        logger.info("Depth transforms initialized")
    
    @staticmethod
    def resize_with_pad(
        image: np.ndarray,
        target_size: Tuple[int, int],
        pad_value: Union[int, float] = 0
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Resize image with padding to maintain aspect ratio.
        
        Args:
            image: Input image.
            target_size: Target size as (width, height).
            pad_value: Value for padding.
            
        Returns:
            Tuple of (resized_image, padding) where padding is (top, bottom, left, right).
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new size
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Apply padding
        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=image.dtype)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        else:
            padded = np.full((target_h, target_w), pad_value, dtype=image.dtype)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        
        return padded, (pad_top, pad_bottom, pad_left, pad_right)
    
    @staticmethod
    def remove_padding(
        image: np.ndarray,
        padding: Tuple[int, int, int, int],
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Remove padding and resize back to original size.
        
        Args:
            image: Padded image.
            padding: Padding as (top, bottom, left, right).
            original_size: Original size as (width, height).
            
        Returns:
            Image resized back to original size.
        """
        pad_top, pad_bottom, pad_left, pad_right = padding
        h, w = image.shape[:2]
        
        # Remove padding
        unpadded_h = h - pad_top - pad_bottom
        unpadded_w = w - pad_left - pad_right
        
        if unpadded_h <= 0 or unpadded_w <= 0:
            # Fallback to direct resize
            return cv2.resize(image, original_size, interpolation=cv2.INTER_LINEAR)
        
        unpadded = image[pad_top:pad_top+unpadded_h, pad_left:pad_left+unpadded_w]
        
        # Resize to original size
        resized = cv2.resize(unpadded, original_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> np.ndarray:
        """Normalize image using ImageNet statistics.
        
        Args:
            image: Input image (RGB, 0-255 or 0-1).
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
            
        Returns:
            Normalized image.
        """
        # Convert to float and normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        normalized = (image - np.array(mean)) / np.array(std)
        
        return normalized.astype(np.float32)
    
    @staticmethod
    def denormalize_image(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> np.ndarray:
        """Denormalize image back to [0, 1] range.
        
        Args:
            image: Normalized image.
            mean: Mean values used for normalization.
            std: Standard deviation values used for normalization.
            
        Returns:
            Denormalized image in [0, 1] range.
        """
        denormalized = image * np.array(std) + np.array(mean)
        return np.clip(denormalized, 0, 1).astype(np.float32)
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array.
        
        Args:
            tensor: Input tensor.
            
        Returns:
            Numpy array.
        """
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        return tensor.numpy()
    
    @staticmethod
    def numpy_to_tensor(
        array: np.ndarray,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.
        
        Args:
            array: Input numpy array.
            device: Target device.
            
        Returns:
            PyTorch tensor.
        """
        tensor = torch.from_numpy(array)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    @staticmethod
    def apply_test_time_augmentation(
        image: np.ndarray,
        model_predict_fn: callable,
        augmentations: list = None
    ) -> np.ndarray:
        """Apply test-time augmentation for improved predictions.
        
        Args:
            image: Input image.
            model_predict_fn: Model prediction function.
            augmentations: List of augmentation types to apply.
            
        Returns:
            Averaged prediction from augmented inputs.
        """
        if augmentations is None:
            augmentations = ["original", "flip_horizontal"]
        
        predictions = []
        
        for aug in augmentations:
            if aug == "original":
                aug_image = image.copy()
            elif aug == "flip_horizontal":
                aug_image = np.fliplr(image)
            elif aug == "flip_vertical":
                aug_image = np.flipud(image)
            else:
                continue
            
            # Get prediction
            pred = model_predict_fn(aug_image)
            
            # Reverse augmentation on prediction
            if aug == "flip_horizontal":
                pred = np.fliplr(pred)
            elif aug == "flip_vertical":
                pred = np.flipud(pred)
            
            predictions.append(pred)
        
        # Average predictions
        if len(predictions) > 0:
            return np.mean(predictions, axis=0)
        else:
            return model_predict_fn(image)
    
    @staticmethod
    def multi_scale_prediction(
        image: np.ndarray,
        model_predict_fn: callable,
        scales: list = [0.5, 1.0, 1.5],
        fusion_method: str = "average"
    ) -> np.ndarray:
        """Apply multi-scale prediction for improved accuracy.
        
        Args:
            image: Input image.
            model_predict_fn: Model prediction function.
            scales: List of scale factors.
            fusion_method: Method to fuse predictions ("average", "median", "max").
            
        Returns:
            Fused prediction from multi-scale inputs.
        """
        original_size = (image.shape[1], image.shape[0])
        predictions = []
        
        for scale in scales:
            # Scale image
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Get prediction
            pred = model_predict_fn(scaled_image)
            
            # Resize prediction back to original size
            pred_resized = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)
            predictions.append(pred_resized)
        
        # Fuse predictions
        predictions = np.array(predictions)
        
        if fusion_method == "average":
            fused = np.mean(predictions, axis=0)
        elif fusion_method == "median":
            fused = np.median(predictions, axis=0)
        elif fusion_method == "max":
            fused = np.max(predictions, axis=0)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        return fused
    
    @staticmethod
    def create_pyramid(
        image: np.ndarray,
        levels: int = 3,
        scale_factor: float = 0.5
    ) -> list:
        """Create image pyramid for multi-resolution processing.
        
        Args:
            image: Input image.
            levels: Number of pyramid levels.
            scale_factor: Scale factor between levels.
            
        Returns:
            List of images at different scales.
        """
        pyramid = [image.copy()]
        
        current_image = image
        for _ in range(levels - 1):
            h, w = current_image.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            if new_h < 32 or new_w < 32:  # Minimum size threshold
                break
            
            current_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pyramid.append(current_image)
        
        return pyramid 