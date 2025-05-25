"""Base class for generative depth estimation models."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaseDepthEstimator(ABC):
    """Abstract base class for depth estimation models."""
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        enable_amp: bool = True
    ):
        """Initialize base depth estimator.
        
        Args:
            device: Device to run inference on. If None, auto-detects best device.
            enable_amp: Whether to enable automatic mixed precision.
        """
        self.device = self._get_device(device)
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.model: Optional[torch.nn.Module] = None
        self.transform = None
        
        logger.info(f"Initialized depth estimator on device: {self.device}")
        if self.enable_amp:
            logger.info("Automatic mixed precision enabled")
    
    def _get_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Get the appropriate device for inference.
        
        Args:
            device: Requested device or None for auto-detection.
            
        Returns:
            PyTorch device object.
        """
        if device is not None:
            return torch.device(device)
        
        # Auto-detect best device
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_name}")
            return torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return torch.device("mps")
        else:
            logger.info("Using CPU")
            return torch.device("cpu")
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load the depth estimation model.
        
        Args:
            model_name: Name or path of the model to load.
            **kwargs: Additional model-specific arguments.
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image for the model.
        
        Args:
            image: Input image as numpy array (RGB).
            
        Returns:
            Preprocessed tensor ready for model inference.
        """
        pass
    
    @abstractmethod
    def postprocess(self, output: torch.Tensor, original_size: tuple) -> np.ndarray:
        """Postprocess model output to depth map.
        
        Args:
            output: Raw model output tensor.
            original_size: Original image size (height, width).
            
        Returns:
            Depth map as numpy array.
        """
        pass
    
    def predict(
        self,
        image: np.ndarray,
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """Predict depth from a single image.
        
        Args:
            image: Input image as numpy array (RGB).
            return_tensor: Whether to return tensor instead of numpy array.
            
        Returns:
            Depth map as numpy array or tensor.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        original_size = image.shape[:2]  # (height, width)
        
        # Preprocess
        input_tensor = self.preprocess(image)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            if self.enable_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
        
        # Postprocess
        if return_tensor:
            return output
        else:
            depth_map = self.postprocess(output, original_size)
            return depth_map
    
    def predict_batch(
        self,
        images: list[np.ndarray],
        batch_size: int = 4
    ) -> list[np.ndarray]:
        """Predict depth for a batch of images.
        
        Args:
            images: List of input images as numpy arrays (RGB).
            batch_size: Batch size for inference.
            
        Returns:
            List of depth maps as numpy arrays.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        depth_maps = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_depths = []
            
            # Process batch
            for image in batch_images:
                depth = self.predict(image)
                batch_depths.append(depth)
            
            depth_maps.extend(batch_depths)
        
        return depth_maps
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "Model loaded",
            "device": str(self.device),
            "amp_enabled": self.enable_amp,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_type": type(self.model).__name__,
        }
    
    def to(self, device: Union[str, torch.device]) -> None:
        """Move model to specified device.
        
        Args:
            device: Target device.
        """
        if self.model is not None:
            self.device = torch.device(device)
            self.model = self.model.to(self.device)
            logger.info(f"Moved model to device: {self.device}")
        else:
            logger.warning("No model loaded to move")
    
    def eval(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
    
    def train(self) -> None:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
    
    def get_memory_usage(self) -> dict:
        """Get GPU memory usage information.
        
        Returns:
            Dictionary with memory usage statistics.
        """
        if not torch.cuda.is_available():
            return {"status": "CUDA not available"}
        
        memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
        max_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3   # GB
        
        return {
            "allocated_gb": round(memory_allocated, 2),
            "reserved_gb": round(memory_reserved, 2),
            "max_allocated_gb": round(max_memory, 2),
            "device": str(self.device),
        } 