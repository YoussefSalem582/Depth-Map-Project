"""Plotting utilities for depth estimation evaluation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class DepthPlotter:
    """Plotting utilities for depth estimation evaluation."""
    
    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        """Initialize depth plotter.
        
        Args:
            style: Matplotlib style to use.
            figsize: Default figure size.
        """
        self.style = style
        self.figsize = figsize
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
        
        logger.info("Depth plotter initialized")
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Method Comparison"
    ) -> plt.Figure:
        """Plot comparison of metrics across methods.
        
        Args:
            results: Dictionary of method results.
            metrics: List of metrics to plot. If None, plots all.
            save_path: Optional path to save the figure.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        if metrics is None:
            metrics = df.columns.tolist()
        
        # Filter metrics
        df_filtered = df[metrics]
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Bar plot
            methods = df_filtered.index.tolist()
            values = df_filtered[metric].values
            
            bars = ax.bar(methods, values, alpha=0.7)
            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel('Value')
            
            # Rotate x-axis labels if needed
            if len(max(methods, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_error_distribution(
        self,
        errors: Dict[str, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Error Distribution"
    ) -> plt.Figure:
        """Plot error distribution for different methods.
        
        Args:
            errors: Dictionary of method errors.
            save_path: Optional path to save the figure.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        for method, error_values in errors.items():
            # Remove invalid values
            valid_errors = error_values[np.isfinite(error_values)]
            
            ax1.hist(valid_errors, bins=50, alpha=0.6, label=method, density=True)
        
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Histogram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        error_data = []
        method_labels = []
        
        for method, error_values in errors.items():
            valid_errors = error_values[np.isfinite(error_values)]
            error_data.append(valid_errors)
            method_labels.append(method)
        
        ax2.boxplot(error_data, labels=method_labels)
        ax2.set_ylabel('Error')
        ax2.set_title('Error Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Rotate labels if needed
        if len(max(method_labels, key=len)) > 8:
            ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_depth_vs_error(
        self,
        depths: np.ndarray,
        errors: np.ndarray,
        method_name: str = "Method",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot error vs depth relationship.
        
        Args:
            depths: Ground truth depth values.
            errors: Error values.
            method_name: Name of the method.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        # Remove invalid values
        valid_mask = np.isfinite(depths) & np.isfinite(errors) & (depths > 0)
        valid_depths = depths[valid_mask]
        valid_errors = errors[valid_mask]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Scatter plot
        ax1.scatter(valid_depths, valid_errors, alpha=0.5, s=1)
        ax1.set_xlabel('Ground Truth Depth (m)')
        ax1.set_ylabel('Error')
        ax1.set_title(f'{method_name}: Error vs Depth')
        ax1.grid(True, alpha=0.3)
        
        # Binned error plot
        depth_bins = np.linspace(valid_depths.min(), valid_depths.max(), 20)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
        bin_errors = []
        
        for i in range(len(depth_bins) - 1):
            mask = (valid_depths >= depth_bins[i]) & (valid_depths < depth_bins[i + 1])
            if np.any(mask):
                bin_errors.append(np.mean(valid_errors[mask]))
            else:
                bin_errors.append(np.nan)
        
        ax2.plot(bin_centers, bin_errors, 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Ground Truth Depth (m)')
        ax2.set_ylabel('Mean Error')
        ax2.set_title(f'{method_name}: Mean Error vs Depth')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Metric Correlation Matrix"
    ) -> plt.Figure:
        """Plot correlation matrix of metrics.
        
        Args:
            results: Dictionary of method results.
            save_path: Optional path to save the figure.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig


def create_comparison_plot(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Method Comparison"
) -> plt.Figure:
    """Convenience function to create a comparison plot.
    
    Args:
        results: Dictionary of method results.
        metrics: List of metrics to plot.
        save_path: Optional path to save the figure.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    plotter = DepthPlotter()
    return plotter.plot_metrics_comparison(results, metrics, save_path, title) 