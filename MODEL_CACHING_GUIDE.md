# 💾 Model Caching & Instant Loading Guide

## 🎯 **Overview**

This guide explains how to save and cache MiDaS depth estimation models for **instant loading**. The system supports multiple formats including `.pt` (PyTorch), `.h5` (HDF5), and `.pkl` (Pickle) for maximum flexibility.

## 🚀 **Quick Start**

### **1. Automatic Setup (Recommended)**
```bash
# Run the setup script to download and cache models
python setup_models.py
```

### **2. Manual Model Management**
```bash
# Download and cache a specific model
python manage_models.py download --model DPT_Large

# Show cache information
python manage_models.py info --detailed

# Export model in .h5 format
python manage_models.py export --model dpt_large --output ./my_model.h5 --format h5
```

## 📋 **Features**

### **✅ Supported Formats**
- **`.pt`** - PyTorch state dict (default, fastest loading)
- **`.h5`** - HDF5 format (cross-platform, widely supported)
- **`.pkl`** - Pickle format (metadata and configuration)

### **✅ Automatic Caching**
- **First-time download**: Model is automatically cached
- **Subsequent loads**: Instant loading from cache
- **Multiple formats**: Models saved in all supported formats
- **Integrity checking**: Corrupted files are automatically detected and removed

### **✅ Cache Management**
- **Smart storage**: Models stored in user cache directory
- **Size optimization**: Duplicate and corrupted files removed
- **Export/Import**: Easy model sharing and backup
- **Statistics**: Detailed cache usage information

## 🔧 **Technical Details**

### **Cache Directory Structure**
```
~/.cache/depthmap/models/
├── midas_dpt_large.pt      # PyTorch format
├── midas_dpt_large.h5      # HDF5 format
├── midas_dpt_large.pkl     # Metadata
├── midas_midas_small.pt    # Small model
└── midas_midas_small.h5    # Small model H5
```

### **Model Loading Priority**
1. **PyTorch cache** (`.pt`) - Fastest
2. **HDF5 cache** (`.h5`) - Cross-platform
3. **Download from hub** - First time only

### **Cache Location**
- **Windows**: `C:\Users\{username}\.cache\depthmap\models\`
- **Linux/Mac**: `~/.cache/depthmap/models/`
- **Custom**: Set via `cache_dir` parameter

## 📖 **Usage Examples**

### **Python API**

#### **Basic Usage with Caching**
```python
from depthmap.generative.midas import MiDaSDepthEstimator

# Initialize with caching enabled (default)
estimator = MiDaSDepthEstimator(
    model_name="DPT_Large",
    use_cache=True  # Enable caching
)

# First run: Downloads and caches model
# Subsequent runs: Loads instantly from cache
depth_map = estimator.predict(image)
```

#### **Custom Cache Directory**
```python
# Use custom cache directory
estimator = MiDaSDepthEstimator(
    model_name="DPT_Large",
    cache_dir="./my_models",
    use_cache=True
)
```

#### **Model Manager API**
```python
from depthmap.utils.model_manager import get_model_manager

# Get model manager instance
manager = get_model_manager()

# Check if model is cached
is_cached = manager.is_model_cached("dpt_large", format="pt")

# Get cache information
cache_info = manager.get_cache_info()
print(f"Total cached models: {len(cache_info['models'])}")

# Export model
manager.export_model("dpt_large", "./exported_model.h5", format="h5")

# Clear specific model
manager.delete_model("dpt_large")

# Optimize cache (remove corrupted files)
results = manager.optimize_cache()
```

### **Command Line Interface**

#### **Download and Cache Models**
```bash
# Download DPT_Large model
python manage_models.py download --model DPT_Large

# Download multiple models
python manage_models.py download --model DPT_Large
python manage_models.py download --model MiDaS_small
```

#### **Cache Management**
```bash
# Show cache information
python manage_models.py info

# Show detailed cache information
python manage_models.py info --detailed

# List cached models
python manage_models.py list

# List models in specific format
python manage_models.py list --format h5
```

#### **Export/Import Models**
```bash
# Export model in PyTorch format
python manage_models.py export --model dpt_large --output ./model.pt

# Export model in HDF5 format
python manage_models.py export --model dpt_large --output ./model.h5 --format h5

# Import external model
python manage_models.py import --input ./external_model.pt --name my_custom_model
```

#### **Cache Optimization**
```bash
# Optimize cache (remove corrupted files)
python manage_models.py optimize

# Clear specific model
python manage_models.py clear --model dpt_large

# Clear entire cache (requires confirmation)
python manage_models.py clear --confirm
```

### **Web API Endpoints**

#### **Cache Information**
```bash
# Get cache info
curl http://localhost:5000/api/models/cache/info

# Get model status
curl http://localhost:5000/api/models/status
```

#### **Cache Management**
```bash
# Clear cache (requires confirmation)
curl -X POST http://localhost:5000/api/models/cache/clear \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'

# Optimize cache
curl -X POST http://localhost:5000/api/models/cache/optimize
```

#### **Model Export**
```bash
# Export model
curl -X POST http://localhost:5000/api/models/export/dpt_large \
  -H "Content-Type: application/json" \
  -d '{"export_path": "./exported_model.h5", "format": "h5"}'
```

## 🎯 **Available Models**

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| **DPT_Large** | ~1.3GB | Highest accuracy | Production, best quality |
| **DPT_Hybrid** | ~800MB | Good balance | General use, good performance |
| **MiDaS** | ~400MB | Standard model | Balanced speed/quality |
| **MiDaS_small** | ~100MB | Fastest | Testing, real-time applications |

## ⚡ **Performance Comparison**

### **Loading Times**

| Scenario | Time | Description |
|----------|------|-------------|
| **First Download** | 30-120s | Download from internet + cache |
| **Cache Load (PT)** | 0.5-2s | Load from PyTorch cache |
| **Cache Load (H5)** | 1-3s | Load from HDF5 cache |
| **No Cache** | 30-120s | Download every time |

### **Storage Requirements**

| Format | Size Overhead | Compatibility |
|--------|---------------|---------------|
| **PT** | 1x | PyTorch only |
| **H5** | 1.1x | Cross-platform |
| **PKL** | 0.01x | Metadata only |

## 🔧 **Advanced Configuration**

### **Custom Cache Settings**
```python
# Initialize with custom settings
estimator = MiDaSDepthEstimator(
    model_name="DPT_Large",
    cache_dir="./custom_cache",  # Custom cache directory
    use_cache=True,              # Enable caching
    optimize=True                # Enable model optimization
)
```

### **Environment Variables**
```bash
# Set custom cache directory
export DEPTHMAP_CACHE_DIR="/path/to/cache"

# Disable caching globally
export DEPTHMAP_DISABLE_CACHE=1
```

### **Programmatic Cache Control**
```python
from depthmap.utils.model_manager import get_model_manager

manager = get_model_manager()

# Get detailed cache statistics
stats = manager.get_cache_stats()
print(f"Cache health: {stats['cache_health']}")
print(f"Total size: {stats['total_size_mb']} MB")

# Optimize cache automatically
if stats['cache_health'] == 'cluttered':
    results = manager.optimize_cache()
    print(f"Freed {results['space_freed_mb']} MB")
```

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Model Not Loading from Cache**
```bash
# Check cache status
python manage_models.py info

# Optimize cache to remove corrupted files
python manage_models.py optimize

# Re-download if necessary
python manage_models.py download --model DPT_Large
```

#### **Cache Directory Issues**
```bash
# Check permissions
ls -la ~/.cache/depthmap/

# Clear and recreate cache
python manage_models.py clear --confirm
python setup_models.py
```

#### **Disk Space Issues**
```bash
# Check cache size
python manage_models.py info

# Clear unused models
python manage_models.py clear --model old_model_name

# Optimize cache
python manage_models.py optimize
```

### **Error Messages**

| Error | Solution |
|-------|----------|
| `Cache directory not writable` | Check permissions on cache directory |
| `Model file corrupted` | Run `python manage_models.py optimize` |
| `Insufficient disk space` | Clear cache or free up disk space |
| `Network timeout` | Check internet connection, retry download |

## 📊 **Cache Monitoring**

### **Cache Health Indicators**
- **Good**: Normal operation, optimal performance
- **Large**: Cache > 5GB, consider cleanup
- **Cluttered**: > 50 files, run optimization

### **Monitoring Commands**
```bash
# Quick status check
python manage_models.py info

# Detailed analysis
python manage_models.py info --detailed

# Cache optimization
python manage_models.py optimize
```

## 🎉 **Benefits**

### **⚡ Performance**
- **10-100x faster** model loading
- **Instant startup** for applications
- **Reduced bandwidth** usage

### **💾 Storage**
- **Multiple formats** for flexibility
- **Automatic optimization** removes waste
- **Smart caching** prevents duplicates

### **🔧 Management**
- **Easy export/import** for model sharing
- **Comprehensive monitoring** and statistics
- **Automated cleanup** and optimization

## 🚀 **Next Steps**

1. **Run setup**: `python setup_models.py`
2. **Test loading**: `python test_enhanced_depth.py`
3. **Start application**: `python app.py`
4. **Monitor cache**: `python manage_models.py info`

Your models are now cached and will load **instantly** every time! 🎉 