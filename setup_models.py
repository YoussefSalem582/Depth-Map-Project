#!/usr/bin/env python3
"""
Setup script for downloading and caching MiDaS models.
This ensures models are available for instant loading.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main setup function."""
    print("🚀 MiDaS Model Setup & Caching")
    print("=" * 50)
    print("This script will download and cache MiDaS models for instant loading.")
    print("Models will be saved in multiple formats (.pt, .h5) for flexibility.\n")
    
    try:
        from depthmap.generative.midas import MiDaSDepthEstimator
        from depthmap.utils.model_manager import get_model_manager
        
        # Initialize model manager
        model_manager = get_model_manager()
        print(f"📁 Cache directory: {model_manager.cache_dir}")
        
        # Available models with descriptions
        models = {
            "DPT_Large": {
                "description": "Highest accuracy, ~1.3GB, best for production",
                "recommended": True
            },
            "DPT_Hybrid": {
                "description": "Good balance, ~800MB, good for most uses",
                "recommended": False
            },
            "MiDaS": {
                "description": "Standard model, ~400MB, good performance",
                "recommended": False
            },
            "MiDaS_small": {
                "description": "Fastest, ~100MB, good for testing",
                "recommended": False
            }
        }
        
        print("📋 Available Models:")
        for model_name, info in models.items():
            status = "✅ RECOMMENDED" if info["recommended"] else "  "
            cached = "🔄 CACHED" if model_manager.is_model_cached(model_name.lower()) else "📥 NOT CACHED"
            print(f"  {status} {model_name}: {info['description']} - {cached}")
        
        print("\n" + "=" * 50)
        
        # Ask user which models to download
        print("Which models would you like to download and cache?")
        print("1. DPT_Large only (recommended)")
        print("2. DPT_Large + MiDaS_small (recommended + fast)")
        print("3. All models (complete setup)")
        print("4. Custom selection")
        print("5. Skip download (just show cache info)")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        models_to_download = []
        
        if choice == "1":
            models_to_download = ["DPT_Large"]
        elif choice == "2":
            models_to_download = ["DPT_Large", "MiDaS_small"]
        elif choice == "3":
            models_to_download = list(models.keys())
        elif choice == "4":
            print("\nSelect models to download (enter numbers separated by spaces):")
            for i, model_name in enumerate(models.keys(), 1):
                print(f"  {i}. {model_name}")
            
            selection = input("Enter selection: ").strip().split()
            model_list = list(models.keys())
            
            for sel in selection:
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(model_list):
                        models_to_download.append(model_list[idx])
                except ValueError:
                    pass
        elif choice == "5":
            models_to_download = []
        else:
            print("Invalid choice, defaulting to DPT_Large only")
            models_to_download = ["DPT_Large"]
        
        # Download selected models
        if models_to_download:
            print(f"\n📥 Downloading {len(models_to_download)} model(s)...")
            print("This may take several minutes depending on your internet connection.\n")
            
            for i, model_name in enumerate(models_to_download, 1):
                print(f"[{i}/{len(models_to_download)}] Downloading {model_name}...")
                
                # Check if already cached
                if model_manager.is_model_cached(model_name.lower()):
                    print(f"  ✅ {model_name} already cached, skipping download")
                    continue
                
                try:
                    start_time = time.time()
                    
                    # Download and cache model
                    estimator = MiDaSDepthEstimator(model_name=model_name, use_cache=True)
                    
                    download_time = time.time() - start_time
                    print(f"  ✅ {model_name} downloaded and cached in {download_time:.1f}s")
                    
                    # Show cache info
                    cache_info = estimator.get_cache_info()
                    if cache_info.get('cached_models'):
                        for cached_model in cache_info['cached_models']:
                            if model_name.lower() in cached_model['name'].lower():
                                print(f"     💾 Cached as: {cached_model['name']} ({cached_model['size_mb']:.1f} MB)")
                    
                except Exception as e:
                    print(f"  ❌ Failed to download {model_name}: {e}")
                    continue
                
                print()
        
        # Show final cache status
        print("📊 Final Cache Status")
        print("=" * 50)
        
        cache_info = model_manager.get_cache_info()
        cache_stats = model_manager.get_cache_stats()
        
        print(f"Total Models Cached: {cache_stats['total_models']}")
        print(f"Total Cache Size: {cache_stats['total_size_mb']:.1f} MB")
        print(f"Cache Directory: {cache_info['cache_dir']}")
        
        if cache_info['models']:
            print("\nCached Models:")
            for model_name, model_info in cache_info['models'].items():
                print(f"  🔹 {model_name}: {model_info['total_size_mb']:.1f} MB")
                for format_name, format_info in model_info['formats'].items():
                    print(f"     {format_name.upper()}: {format_info['size_mb']:.1f} MB")
        
        print("\n🎉 Setup Complete!")
        print("=" * 50)
        print("Your models are now cached and will load instantly!")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the web application")
        print("2. Use 'python manage_models.py info' to view cache details")
        print("3. Use 'python manage_models.py export' to export models")
        
        # Test model loading speed
        if cache_stats['total_models'] > 0:
            print("\n⚡ Testing model loading speed...")
            try:
                # Find a cached model to test
                test_model = None
                for model_name in cache_info['models'].keys():
                    if 'dpt_large' in model_name.lower():
                        test_model = 'DPT_Large'
                        break
                    elif 'midas' in model_name.lower():
                        test_model = 'MiDaS'
                        break
                
                if test_model:
                    start_time = time.time()
                    test_estimator = MiDaSDepthEstimator(model_name=test_model, use_cache=True)
                    load_time = time.time() - start_time
                    print(f"✅ {test_model} loaded from cache in {load_time:.2f}s (instant!)")
                
            except Exception as e:
                print(f"⚠️  Model loading test failed: {e}")
        
    except ImportError as e:
        print(f"❌ Error: Required dependencies not available: {e}")
        print("\nPlease install required packages first:")
        print("pip install -r requirements.txt")
        return
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        return
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return

if __name__ == "__main__":
    main() 