"""Depth estimation evaluation metrics."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DepthMetrics:
    """Comprehensive depth estimation evaluation metrics."""
    
    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        depth_cap: Optional[float] = None
    ):
        """Initialize depth metrics calculator.
        
        Args:
            min_depth: Minimum valid depth value.
            max_depth: Maximum valid depth value.
            depth_cap: Optional depth cap for evaluation (e.g., KITTI uses 80m).
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_cap = depth_cap or max_depth
        
        logger.debug(f"Initialized depth metrics with range [{min_depth}, {max_depth}]")
    
    def _get_valid_mask(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray
    ) -> np.ndarray:
        """Get mask for valid depth values.
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            
        Returns:
            Boolean mask for valid pixels.
        """
        valid_mask = (
            (gt_depth > self.min_depth) &
            (gt_depth < self.depth_cap) &
            (pred_depth > self.min_depth) &
            (pred_depth < self.max_depth) &
            np.isfinite(gt_depth) &
            np.isfinite(pred_depth)
        )
        return valid_mask
    
    def rmse(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        log_scale: bool = False
    ) -> float:
        """Root Mean Square Error.
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            log_scale: Whether to compute RMSE in log scale.
            
        Returns:
            RMSE value.
        """
        valid_mask = self._get_valid_mask(pred_depth, gt_depth)
        
        if not np.any(valid_mask):
            logger.warning("No valid pixels found for RMSE computation")
            return float('inf')
        
        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        
        if log_scale:
            # RMSE in log scale
            log_pred = np.log(pred_valid)
            log_gt = np.log(gt_valid)
            rmse_val = np.sqrt(np.mean((log_pred - log_gt) ** 2))
        else:
            # Standard RMSE
            rmse_val = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
        
        return float(rmse_val)
    
    def mae(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        log_scale: bool = False
    ) -> float:
        """Mean Absolute Error.
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            log_scale: Whether to compute MAE in log scale.
            
        Returns:
            MAE value.
        """
        valid_mask = self._get_valid_mask(pred_depth, gt_depth)
        
        if not np.any(valid_mask):
            logger.warning("No valid pixels found for MAE computation")
            return float('inf')
        
        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        
        if log_scale:
            # MAE in log scale
            log_pred = np.log(pred_valid)
            log_gt = np.log(gt_valid)
            mae_val = np.mean(np.abs(log_pred - log_gt))
        else:
            # Standard MAE
            mae_val = np.mean(np.abs(pred_valid - gt_valid))
        
        return float(mae_val)
    
    def silog(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray
    ) -> float:
        """Scale-Invariant Logarithmic Error (SILog).
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            
        Returns:
            SILog value.
        """
        valid_mask = self._get_valid_mask(pred_depth, gt_depth)
        
        if not np.any(valid_mask):
            logger.warning("No valid pixels found for SILog computation")
            return float('inf')
        
        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        
        # Compute log difference
        log_diff = np.log(pred_valid) - np.log(gt_valid)
        
        # SILog formula
        silog_val = np.sqrt(
            np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2)
        ) * 100  # Convert to percentage
        
        return float(silog_val)
    
    def delta_threshold(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        threshold: float = 1.25
    ) -> float:
        """Delta threshold accuracy (δ < threshold).
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            threshold: Threshold value (typically 1.25, 1.25^2, 1.25^3).
            
        Returns:
            Percentage of pixels within threshold.
        """
        valid_mask = self._get_valid_mask(pred_depth, gt_depth)
        
        if not np.any(valid_mask):
            logger.warning("No valid pixels found for delta threshold computation")
            return 0.0
        
        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        
        # Compute ratio
        ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        
        # Count pixels within threshold
        within_threshold = np.sum(ratio < threshold)
        total_pixels = len(ratio)
        
        accuracy = (within_threshold / total_pixels) * 100
        
        return float(accuracy)
    
    def bad_pixel_ratio(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        threshold: float = 1.0
    ) -> float:
        """Bad pixel ratio (percentage of pixels with error > threshold).
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            threshold: Error threshold in meters.
            
        Returns:
            Percentage of bad pixels.
        """
        valid_mask = self._get_valid_mask(pred_depth, gt_depth)
        
        if not np.any(valid_mask):
            logger.warning("No valid pixels found for bad pixel ratio computation")
            return 100.0
        
        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        
        # Compute absolute error
        abs_error = np.abs(pred_valid - gt_valid)
        
        # Count bad pixels
        bad_pixels = np.sum(abs_error > threshold)
        total_pixels = len(abs_error)
        
        bad_ratio = (bad_pixels / total_pixels) * 100
        
        return float(bad_ratio)
    
    def relative_error(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        threshold: float = 0.1
    ) -> float:
        """Relative error (percentage of pixels with relative error > threshold).
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            threshold: Relative error threshold (e.g., 0.1 for 10%).
            
        Returns:
            Percentage of pixels with high relative error.
        """
        valid_mask = self._get_valid_mask(pred_depth, gt_depth)
        
        if not np.any(valid_mask):
            logger.warning("No valid pixels found for relative error computation")
            return 100.0
        
        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]
        
        # Compute relative error
        rel_error = np.abs(pred_valid - gt_valid) / gt_valid
        
        # Count pixels with high relative error
        high_error_pixels = np.sum(rel_error > threshold)
        total_pixels = len(rel_error)
        
        high_error_ratio = (high_error_pixels / total_pixels) * 100
        
        return float(high_error_ratio)
    
    def compute_all_metrics(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray
    ) -> Dict[str, float]:
        """Compute all depth evaluation metrics.
        
        Args:
            pred_depth: Predicted depth map.
            gt_depth: Ground truth depth map.
            
        Returns:
            Dictionary containing all metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics['rmse'] = self.rmse(pred_depth, gt_depth)
        metrics['rmse_log'] = self.rmse(pred_depth, gt_depth, log_scale=True)
        metrics['mae'] = self.mae(pred_depth, gt_depth)
        metrics['mae_log'] = self.mae(pred_depth, gt_depth, log_scale=True)
        metrics['silog'] = self.silog(pred_depth, gt_depth)
        
        # Delta thresholds
        metrics['delta1'] = self.delta_threshold(pred_depth, gt_depth, 1.25)
        metrics['delta2'] = self.delta_threshold(pred_depth, gt_depth, 1.25**2)
        metrics['delta3'] = self.delta_threshold(pred_depth, gt_depth, 1.25**3)
        
        # Bad pixel ratios
        metrics['bad_1.0'] = self.bad_pixel_ratio(pred_depth, gt_depth, 1.0)
        metrics['bad_2.0'] = self.bad_pixel_ratio(pred_depth, gt_depth, 2.0)
        metrics['bad_3.0'] = self.bad_pixel_ratio(pred_depth, gt_depth, 3.0)
        
        # Relative errors
        metrics['rel_10%'] = self.relative_error(pred_depth, gt_depth, 0.1)
        metrics['rel_20%'] = self.relative_error(pred_depth, gt_depth, 0.2)
        metrics['rel_30%'] = self.relative_error(pred_depth, gt_depth, 0.3)
        
        # Additional statistics
        valid_mask = self._get_valid_mask(pred_depth, gt_depth)
        metrics['valid_pixels'] = int(np.sum(valid_mask))
        metrics['total_pixels'] = int(pred_depth.size)
        metrics['valid_ratio'] = float(np.sum(valid_mask) / pred_depth.size * 100)
        
        return metrics


def compute_all_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
    depth_cap: Optional[float] = None
) -> Dict[str, float]:
    """Convenience function to compute all depth metrics.
    
    Args:
        pred_depth: Predicted depth map.
        gt_depth: Ground truth depth map.
        min_depth: Minimum valid depth value.
        max_depth: Maximum valid depth value.
        depth_cap: Optional depth cap for evaluation.
        
    Returns:
        Dictionary containing all metrics.
    """
    metrics_calculator = DepthMetrics(min_depth, max_depth, depth_cap)
    return metrics_calculator.compute_all_metrics(pred_depth, gt_depth)


def compare_methods(
    predictions: Dict[str, np.ndarray],
    gt_depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 80.0
) -> Dict[str, Dict[str, float]]:
    """Compare multiple depth estimation methods.
    
    Args:
        predictions: Dictionary mapping method names to predicted depth maps.
        gt_depth: Ground truth depth map.
        min_depth: Minimum valid depth value.
        max_depth: Maximum valid depth value.
        
    Returns:
        Dictionary mapping method names to their metrics.
    """
    results = {}
    
    for method_name, pred_depth in predictions.items():
        logger.info(f"Evaluating method: {method_name}")
        metrics = compute_all_metrics(pred_depth, gt_depth, min_depth, max_depth)
        results[method_name] = metrics
    
    return results


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    metrics_to_show: Optional[list[str]] = None
) -> None:
    """Print metrics comparison table.
    
    Args:
        results: Dictionary mapping method names to their metrics.
        metrics_to_show: List of metrics to display. If None, shows key metrics.
    """
    if metrics_to_show is None:
        metrics_to_show = ['rmse', 'mae', 'silog', 'delta1', 'delta2', 'delta3', 'bad_1.0']
    
    # Print header
    print(f"{'Method':<15}", end="")
    for metric in metrics_to_show:
        print(f"{metric:>10}", end="")
    print()
    
    print("-" * (15 + 10 * len(metrics_to_show)))
    
    # Print results
    for method_name, metrics in results.items():
        print(f"{method_name:<15}", end="")
        for metric in metrics_to_show:
            value = metrics.get(metric, 0.0)
            if metric.startswith('delta') or metric.startswith('rel_'):
                print(f"{value:>9.1f}%", end="")
            else:
                print(f"{value:>9.3f}", end="")
        print()


def get_best_method(
    results: Dict[str, Dict[str, float]],
    metric: str = 'rmse',
    lower_is_better: bool = True
) -> Tuple[str, float]:
    """Find the best performing method for a given metric.
    
    Args:
        results: Dictionary mapping method names to their metrics.
        metric: Metric to use for comparison.
        lower_is_better: Whether lower values are better for this metric.
        
    Returns:
        Tuple of (best_method_name, best_value).
    """
    if not results:
        raise ValueError("No results provided")
    
    best_method = None
    best_value = float('inf') if lower_is_better else float('-inf')
    
    for method_name, metrics in results.items():
        if metric not in metrics:
            logger.warning(f"Metric '{metric}' not found for method '{method_name}'")
            continue
        
        value = metrics[metric]
        
        if lower_is_better:
            if value < best_value:
                best_value = value
                best_method = method_name
        else:
            if value > best_value:
                best_value = value
                best_method = method_name
    
    if best_method is None:
        raise ValueError(f"Could not find best method for metric '{metric}'")
    
    return best_method, best_value 