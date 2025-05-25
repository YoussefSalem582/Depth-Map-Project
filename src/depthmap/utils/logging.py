"""Logging utilities for the depth estimation project."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "depthmap",
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """Setup logger with console and/or file output.
    
    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_string: Custom format string for log messages.
        console_output: Whether to output to console.
        
    Returns:
        Configured logger instance.
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "depthmap") -> logging.Logger:
    """Get existing logger or create a basic one.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic configuration
    if not logger.handlers:
        setup_logger(name)
    
    return logger


def set_log_level(level: Union[str, int], logger_name: str = "depthmap") -> None:
    """Set logging level for specified logger.
    
    Args:
        level: New logging level.
        logger_name: Name of logger to modify.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """Log system information for debugging.
    
    Args:
        logger: Logger instance. If None, uses default logger.
    """
    if logger is None:
        logger = get_logger()
    
    import platform
    import torch
    import cv2
    import numpy as np
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA: Not available")
    
    # MPS info (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon): Available")
    
    logger.info("=== End System Information ===")


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        
        return super().format(record)


def setup_colored_logger(
    name: str = "depthmap",
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """Setup logger with colored console output.
    
    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path to log file.
        
    Returns:
        Configured logger with colored output.
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Colored console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    colored_formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    
    # File handler (without colors)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger 