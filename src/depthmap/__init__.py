"""
Depth Map Project: Comprehensive depth estimation comparison.

This package provides tools for comparing classical stereo vision depth estimation
with modern monocular generative models using large-scale datasets.
"""

__version__ = "0.1.0"
__author__ = "Computer Vision Team"
__email__ = "team@depthmap.ai"

from . import classical, generative, eval, datasets, utils

__all__ = ["classical", "generative", "eval", "datasets", "utils", "__version__"] 