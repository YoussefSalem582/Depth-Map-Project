"""MiDaS monocular depth estimation implementation."""

import logging
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .base import BaseDepthEstimator

logger = logging.getLogger(__name__)


class MiDaSDepthEstimator(BaseDepthEstimator):
    """MiDaS monocular depth estimation using pretrained models."""
    
    # Available MiDaS models
    AVAILABLE_MODELS = {
        "DPT_Large": "intel-isl/MiDaS",
        "DPT_Hybrid": "intel-isl/MiDaS", 
        "MiDaS": "intel-isl/MiDaS",
        "MiDaS_small": "intel-isl/MiDaS",
    }
    
    # Model input sizes
    MODEL_SIZES = {
        "DPT_Large": (384, 384),
        "DPT_Hybrid": (384, 384),
        "MiDaS": (384, 384),
        "MiDaS_small": (256, 256),
    }
    
    def __init__(
        self,
        model_name: str = "DPT_Large",
        device: Optional[Union[str, torch.device]] = None,
        enable_amp: bool = True,
        optimize: bool = True
    ):
        """Initialize MiDaS depth estimator.
        
        Args:
            model_name: Name of the MiDaS model to use.
            device: Device to run inference on.
            enable_amp: Whether to enable automatic mixed precision.
            optimize: Whether to optimize model for inference.
        """
        super().__init__(device, enable_amp)
        
        self.model_name = model_name
        self.optimize = optimize
        self.input_size = self.MODEL_SIZES.get(model_name, (384, 384))
        
        # Load model
        self.load_model(model_name)
        
        logger.info(f"Initialized MiDaS {model_name} depth estimator")
    
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load MiDaS model from torch hub.
        
        Args:
            model_name: Name of the MiDaS model.
            **kwargs: Additional arguments (unused for MiDaS).
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        logger.info(f"Loading MiDaS model: {model_name}")
        
        try:
            # Load model from torch hub
            self.model = torch.hub.load(
                self.AVAILABLE_MODELS[model_name],
                model_name,
                pretrained=True,
                trust_repo=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            if self.optimize:
                self._optimize_model()
            
            # Setup transforms
            self._setup_transforms()
            
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _optimize_model(self) -> None:
        """Optimize model for inference."""
        try:
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                logger.info("Model compiled for faster inference")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        # MiDaS normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        logger.debug(f"Setup transforms for input size: {self.input_size}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image for MiDaS.
        
        Args:
            image: Input image as numpy array (RGB, 0-255).
            
        Returns:
            Preprocessed tensor ready for model inference.
        """
        # Ensure image is uint8 RGB
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Apply transforms
        input_tensor = self.transform(image)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    def postprocess(self, output: torch.Tensor, original_size: tuple) -> np.ndarray:
        """Postprocess MiDaS output to depth map.
        
        Args:
            output: Raw model output tensor.
            original_size: Original image size (height, width).
            
        Returns:
            Depth map as numpy array.
        """
        # Remove batch dimension and move to CPU
        depth = output.squeeze().cpu().numpy()
        
        # Resize to original size
        if depth.shape != original_size:
            depth = cv2.resize(depth, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # MiDaS outputs inverse depth, convert to depth
        # Normalize to [0, 1] range first
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)
        
        # Invert to get actual depth (closer objects have smaller values in MiDaS output)
        depth_map = 1.0 - depth_normalized
        
        return depth_map.astype(np.float32)
    
    def predict_with_confidence(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Predict depth with confidence estimation.
        
        Args:
            image: Input image as numpy array (RGB).
            return_confidence: Whether to return confidence map.
            
        Returns:
            Depth map or tuple of (depth_map, confidence_map).
        """
        depth_map = self.predict(image)
        
        if not return_confidence:
            return depth_map
        
        # Simple confidence estimation based on depth gradient
        # Areas with high gradient are less confident
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Invert gradient to get confidence (low gradient = high confidence)
        confidence = 1.0 / (1.0 + gradient_magnitude)
        confidence = confidence.astype(np.float32)
        
        return depth_map, confidence
    
    def predict_multiscale(
        self,
        image: np.ndarray,
        scales: list[float] = [0.5, 1.0, 1.5],
        fusion_method: str = "average"
    ) -> np.ndarray:
        """Predict depth using multiple scales and fuse results.
        
        Args:
            image: Input image as numpy array (RGB).
            scales: List of scale factors to use.
            fusion_method: Method to fuse multi-scale results ('average', 'median').
            
        Returns:
            Fused depth map as numpy array.
        """
        original_size = image.shape[:2]
        depth_maps = []
        
        for scale in scales:
            # Resize image
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Predict depth
            depth = self.predict(scaled_image)
            
            # Resize depth back to original size
            depth_resized = cv2.resize(depth, (original_size[1], original_size[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            depth_maps.append(depth_resized)
        
        # Fuse depth maps
        depth_stack = np.stack(depth_maps, axis=0)
        
        if fusion_method == "average":
            fused_depth = np.mean(depth_stack, axis=0)
        elif fusion_method == "median":
            fused_depth = np.median(depth_stack, axis=0)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        return fused_depth.astype(np.float32)
    
    def get_supported_models(self) -> list[str]:
        """Get list of supported MiDaS models.
        
        Returns:
            List of model names.
        """
        return list(self.AVAILABLE_MODELS.keys())
    
    def switch_model(self, model_name: str) -> None:
        """Switch to a different MiDaS model.
        
        Args:
            model_name: Name of the new model to load.
        """
        if model_name == self.model_name:
            logger.info(f"Model {model_name} already loaded")
            return
        
        logger.info(f"Switching from {self.model_name} to {model_name}")
        
        # Clear current model
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load new model
        self.model_name = model_name
        self.input_size = self.MODEL_SIZES.get(model_name, (384, 384))
        self.load_model(model_name)
    
    def benchmark_inference(
        self,
        image: np.ndarray,
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> dict:
        """Benchmark inference speed.
        
        Args:
            image: Test image for benchmarking.
            num_runs: Number of inference runs for timing.
            warmup_runs: Number of warmup runs (not timed).
            
        Returns:
            Dictionary with timing statistics.
        """
        import time
        
        logger.info(f"Benchmarking inference with {num_runs} runs")
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.predict(image)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predict(image)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            "model": self.model_name,
            "device": str(self.device),
            "image_size": image.shape[:2],
            "input_size": self.input_size,
            "num_runs": num_runs,
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "min_time_s": float(np.min(times)),
            "max_time_s": float(np.max(times)),
            "fps": float(1.0 / np.mean(times)),
        }
        
        logger.info(f"Benchmark results: {stats['fps']:.2f} FPS (mean: {stats['mean_time_s']:.3f}s)")
        
        return stats 