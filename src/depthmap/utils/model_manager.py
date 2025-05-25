"""Model management utilities for caching and loading depth estimation models."""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model caching and loading for depth estimation models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model manager.
        
        Args:
            cache_dir: Directory to store cached models. If None, uses default.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "depthmap", "models")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model manager initialized with cache dir: {self.cache_dir}")
    
    def get_cache_info(self) -> Dict:
        """Get comprehensive information about cached models.
        
        Returns:
            Dictionary with detailed cache information.
        """
        cache_info = {
            'cache_dir': str(self.cache_dir),
            'models': {},
            'total_size_mb': 0,
            'total_files': 0,
            'formats': {'pt': 0, 'h5': 0, 'pkl': 0, 'onnx': 0}
        }
        
        if not self.cache_dir.exists():
            return cache_info
        
        # Group files by model name
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                cache_info['total_size_mb'] += size_mb
                cache_info['total_files'] += 1
                
                # Extract model name and format
                name_parts = file_path.stem.split('_')
                if len(name_parts) >= 2:
                    model_name = '_'.join(name_parts[1:])  # Remove 'midas' prefix
                    format_ext = file_path.suffix[1:]  # Remove dot
                    
                    if model_name not in cache_info['models']:
                        cache_info['models'][model_name] = {
                            'formats': {},
                            'total_size_mb': 0,
                            'last_modified': None
                        }
                    
                    cache_info['models'][model_name]['formats'][format_ext] = {
                        'size_mb': round(size_mb, 2),
                        'path': str(file_path),
                        'last_modified': time.ctime(file_path.stat().st_mtime)
                    }
                    cache_info['models'][model_name]['total_size_mb'] += size_mb
                    
                    # Update format counts
                    if format_ext in cache_info['formats']:
                        cache_info['formats'][format_ext] += 1
                    
                    # Update last modified
                    file_mtime = file_path.stat().st_mtime
                    if (cache_info['models'][model_name]['last_modified'] is None or 
                        file_mtime > time.mktime(time.strptime(
                            cache_info['models'][model_name]['last_modified']))):
                        cache_info['models'][model_name]['last_modified'] = time.ctime(file_mtime)
        
        # Round total size
        cache_info['total_size_mb'] = round(cache_info['total_size_mb'], 2)
        
        # Round model sizes
        for model_info in cache_info['models'].values():
            model_info['total_size_mb'] = round(model_info['total_size_mb'], 2)
        
        return cache_info
    
    def list_cached_models(self) -> List[str]:
        """Get list of cached model names.
        
        Returns:
            List of model names that have cached files.
        """
        cache_info = self.get_cache_info()
        return list(cache_info['models'].keys())
    
    def is_model_cached(self, model_name: str, format: str = "pt") -> bool:
        """Check if a model is cached in specified format.
        
        Args:
            model_name: Name of the model.
            format: Format to check ('pt', 'h5', 'pkl', 'onnx').
            
        Returns:
            True if model is cached in specified format.
        """
        cache_path = self.cache_dir / f"midas_{model_name.lower()}.{format}"
        return cache_path.exists()
    
    def get_model_path(self, model_name: str, format: str = "pt") -> Optional[Path]:
        """Get path to cached model file.
        
        Args:
            model_name: Name of the model.
            format: Format to get ('pt', 'h5', 'pkl', 'onnx').
            
        Returns:
            Path to model file if it exists, None otherwise.
        """
        cache_path = self.cache_dir / f"midas_{model_name.lower()}.{format}"
        return cache_path if cache_path.exists() else None
    
    def delete_model(self, model_name: str, format: Optional[str] = None) -> bool:
        """Delete cached model files.
        
        Args:
            model_name: Name of the model to delete.
            format: Specific format to delete. If None, deletes all formats.
            
        Returns:
            True if deletion was successful.
        """
        try:
            deleted_files = 0
            
            if format is not None:
                # Delete specific format
                cache_path = self.cache_dir / f"midas_{model_name.lower()}.{format}"
                if cache_path.exists():
                    cache_path.unlink()
                    deleted_files += 1
                    logger.info(f"Deleted {cache_path}")
            else:
                # Delete all formats for this model
                pattern = f"midas_{model_name.lower()}.*"
                for file_path in self.cache_dir.glob(pattern):
                    file_path.unlink()
                    deleted_files += 1
                    logger.info(f"Deleted {file_path}")
            
            logger.info(f"Deleted {deleted_files} files for model {model_name}")
            return deleted_files > 0
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def clear_cache(self, confirm: bool = False) -> bool:
        """Clear all cached models.
        
        Args:
            confirm: Must be True to actually clear cache (safety measure).
            
        Returns:
            True if cache was cleared successfully.
        """
        if not confirm:
            logger.warning("Cache clear not confirmed. Set confirm=True to actually clear.")
            return False
        
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all cached models")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def optimize_cache(self) -> Dict:
        """Optimize cache by removing duplicate or corrupted files.
        
        Returns:
            Dictionary with optimization results.
        """
        results = {
            'files_checked': 0,
            'files_removed': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        try:
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    results['files_checked'] += 1
                    
                    # Check if file is corrupted (for PyTorch files)
                    if file_path.suffix == '.pt':
                        try:
                            torch.load(file_path, map_location='cpu')
                        except Exception as e:
                            # File is corrupted, remove it
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            file_path.unlink()
                            results['files_removed'] += 1
                            results['space_freed_mb'] += size_mb
                            results['errors'].append(f"Removed corrupted file {file_path.name}: {e}")
                            logger.warning(f"Removed corrupted file: {file_path}")
            
            results['space_freed_mb'] = round(results['space_freed_mb'], 2)
            logger.info(f"Cache optimization complete: {results}")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def export_model(self, model_name: str, export_path: str, format: str = "pt") -> bool:
        """Export a cached model to a different location.
        
        Args:
            model_name: Name of the model to export.
            export_path: Path to export the model to.
            format: Format to export ('pt', 'h5', 'pkl').
            
        Returns:
            True if export was successful.
        """
        try:
            source_path = self.get_model_path(model_name, format)
            if source_path is None:
                logger.error(f"Model {model_name} not found in format {format}")
                return False
            
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_path, export_path)
            logger.info(f"Exported {model_name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model {model_name}: {e}")
            return False
    
    def import_model(self, model_path: str, model_name: str) -> bool:
        """Import a model file into the cache.
        
        Args:
            model_path: Path to the model file to import.
            model_name: Name to give the imported model.
            
        Returns:
            True if import was successful.
        """
        try:
            source_path = Path(model_path)
            if not source_path.exists():
                logger.error(f"Source model file not found: {model_path}")
                return False
            
            format_ext = source_path.suffix[1:]  # Remove dot
            target_path = self.cache_dir / f"midas_{model_name.lower()}.{format_ext}"
            
            shutil.copy2(source_path, target_path)
            logger.info(f"Imported model {model_name} from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import model: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics and health information.
        
        Returns:
            Dictionary with cache statistics.
        """
        cache_info = self.get_cache_info()
        
        stats = {
            'total_models': len(cache_info['models']),
            'total_size_mb': cache_info['total_size_mb'],
            'total_files': cache_info['total_files'],
            'formats': cache_info['formats'],
            'average_model_size_mb': 0,
            'largest_model': None,
            'smallest_model': None,
            'cache_health': 'good'
        }
        
        if stats['total_models'] > 0:
            stats['average_model_size_mb'] = round(
                cache_info['total_size_mb'] / stats['total_models'], 2
            )
            
            # Find largest and smallest models
            model_sizes = [(name, info['total_size_mb']) 
                          for name, info in cache_info['models'].items()]
            model_sizes.sort(key=lambda x: x[1])
            
            if model_sizes:
                stats['smallest_model'] = {'name': model_sizes[0][0], 'size_mb': model_sizes[0][1]}
                stats['largest_model'] = {'name': model_sizes[-1][0], 'size_mb': model_sizes[-1][1]}
        
        # Determine cache health
        if cache_info['total_size_mb'] > 5000:  # > 5GB
            stats['cache_health'] = 'large'
        elif stats['total_files'] > 50:
            stats['cache_health'] = 'cluttered'
        
        return stats


# Global model manager instance
_model_manager = None

def get_model_manager(cache_dir: Optional[str] = None) -> ModelManager:
    """Get global model manager instance.
    
    Args:
        cache_dir: Cache directory (only used on first call).
        
    Returns:
        ModelManager instance.
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(cache_dir)
    return _model_manager 