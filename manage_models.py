#!/usr/bin/env python3
"""
Command-line utility for managing depth estimation models and cache.
Provides easy access to model caching, loading, and management operations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Manage depth estimation models and cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show cache information
  python manage_models.py info
  
  # Download and cache MiDaS model
  python manage_models.py download --model DPT_Large
  
  # Export cached model
  python manage_models.py export --model dpt_large --output ./models/midas_large.pt
  
  # Clear cache
  python manage_models.py clear --confirm
  
  # Optimize cache
  python manage_models.py optimize
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show cache information')
    info_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download and cache a model')
    download_parser.add_argument('--model', required=True, 
                               choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS', 'MiDaS_small'],
                               help='Model to download')
    download_parser.add_argument('--format', choices=['pt', 'h5'], default='pt',
                               help='Format to save model in')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export cached model')
    export_parser.add_argument('--model', required=True, help='Model name to export')
    export_parser.add_argument('--output', required=True, help='Output path')
    export_parser.add_argument('--format', choices=['pt', 'h5', 'pkl'], default='pt',
                             help='Format to export')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import model to cache')
    import_parser.add_argument('--input', required=True, help='Input model file path')
    import_parser.add_argument('--name', required=True, help='Name for imported model')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear model cache')
    clear_parser.add_argument('--confirm', action='store_true', 
                            help='Confirm cache clearing (required)')
    clear_parser.add_argument('--model', help='Clear specific model only')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize cache')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List cached models')
    list_parser.add_argument('--format', help='Filter by format')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        from depthmap.utils.model_manager import get_model_manager
        model_manager = get_model_manager()
        
        if args.command == 'info':
            show_cache_info(model_manager, args.detailed)
        elif args.command == 'download':
            download_model(args.model, args.format)
        elif args.command == 'export':
            export_model(model_manager, args.model, args.output, args.format)
        elif args.command == 'import':
            import_model(model_manager, args.input, args.name)
        elif args.command == 'clear':
            clear_cache(model_manager, args.confirm, args.model)
        elif args.command == 'optimize':
            optimize_cache(model_manager)
        elif args.command == 'list':
            list_models(model_manager, args.format)
            
    except ImportError as e:
        print(f"❌ Error: Required dependencies not available: {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_cache_info(model_manager, detailed=False):
    """Show cache information."""
    print("📊 Model Cache Information")
    print("=" * 50)
    
    cache_info = model_manager.get_cache_info()
    cache_stats = model_manager.get_cache_stats()
    
    print(f"Cache Directory: {cache_info['cache_dir']}")
    print(f"Total Models: {cache_stats['total_models']}")
    print(f"Total Size: {cache_stats['total_size_mb']:.2f} MB")
    print(f"Total Files: {cache_stats['total_files']}")
    print(f"Cache Health: {cache_stats['cache_health']}")
    
    if cache_stats['total_models'] > 0:
        print(f"Average Model Size: {cache_stats['average_model_size_mb']:.2f} MB")
        
        if cache_stats['largest_model']:
            print(f"Largest Model: {cache_stats['largest_model']['name']} ({cache_stats['largest_model']['size_mb']:.2f} MB)")
        if cache_stats['smallest_model']:
            print(f"Smallest Model: {cache_stats['smallest_model']['name']} ({cache_stats['smallest_model']['size_mb']:.2f} MB)")
    
    print("\nFormat Distribution:")
    for format_name, count in cache_stats['formats'].items():
        if count > 0:
            print(f"  {format_name.upper()}: {count} files")
    
    if detailed and cache_info['models']:
        print("\n📋 Detailed Model Information:")
        print("-" * 50)
        for model_name, model_info in cache_info['models'].items():
            print(f"\n🔹 {model_name}")
            print(f"   Size: {model_info['total_size_mb']:.2f} MB")
            print(f"   Last Modified: {model_info['last_modified']}")
            print("   Formats:")
            for format_name, format_info in model_info['formats'].items():
                print(f"     {format_name.upper()}: {format_info['size_mb']:.2f} MB")

def download_model(model_name, format_type):
    """Download and cache a model."""
    print(f"📥 Downloading {model_name} model...")
    
    try:
        from depthmap.generative.midas import MiDaSDepthEstimator
        
        # Create estimator which will download and cache the model
        estimator = MiDaSDepthEstimator(model_name=model_name, use_cache=True)
        
        print(f"✅ Successfully downloaded and cached {model_name}")
        
        # Show cache info for this model
        cache_info = estimator.get_cache_info()
        print(f"📊 Cache Info: {json.dumps(cache_info, indent=2)}")
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")

def export_model(model_manager, model_name, output_path, format_type):
    """Export a cached model."""
    print(f"📤 Exporting {model_name} to {output_path}...")
    
    if not model_manager.is_model_cached(model_name, format_type):
        print(f"❌ Model {model_name} not found in {format_type} format")
        available_models = model_manager.list_cached_models()
        if available_models:
            print(f"Available models: {', '.join(available_models)}")
        return
    
    success = model_manager.export_model(model_name, output_path, format_type)
    
    if success:
        print(f"✅ Successfully exported {model_name} to {output_path}")
        
        # Show file size
        output_file = Path(output_path)
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"📊 File size: {size_mb:.2f} MB")
    else:
        print(f"❌ Failed to export {model_name}")

def import_model(model_manager, input_path, model_name):
    """Import a model to cache."""
    print(f"📥 Importing {input_path} as {model_name}...")
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    success = model_manager.import_model(input_path, model_name)
    
    if success:
        print(f"✅ Successfully imported {model_name}")
        
        # Show imported file info
        size_mb = input_file.stat().st_size / (1024 * 1024)
        print(f"📊 Imported size: {size_mb:.2f} MB")
    else:
        print(f"❌ Failed to import {model_name}")

def clear_cache(model_manager, confirm, specific_model=None):
    """Clear model cache."""
    if specific_model:
        print(f"🗑️  Clearing cache for {specific_model}...")
        success = model_manager.delete_model(specific_model)
        if success:
            print(f"✅ Successfully cleared cache for {specific_model}")
        else:
            print(f"❌ Failed to clear cache for {specific_model}")
    else:
        if not confirm:
            print("❌ Cache clear requires confirmation. Use --confirm flag.")
            print("⚠️  This will delete ALL cached models!")
            return
        
        print("🗑️  Clearing entire model cache...")
        success = model_manager.clear_cache(confirm=True)
        
        if success:
            print("✅ Successfully cleared all cached models")
        else:
            print("❌ Failed to clear cache")

def optimize_cache(model_manager):
    """Optimize model cache."""
    print("🔧 Optimizing model cache...")
    
    results = model_manager.optimize_cache()
    
    print(f"📊 Optimization Results:")
    print(f"   Files Checked: {results['files_checked']}")
    print(f"   Files Removed: {results['files_removed']}")
    print(f"   Space Freed: {results['space_freed_mb']:.2f} MB")
    
    if results['errors']:
        print("⚠️  Errors encountered:")
        for error in results['errors']:
            print(f"   {error}")
    
    if results['files_removed'] == 0:
        print("✅ Cache is already optimized")
    else:
        print(f"✅ Cache optimization complete")

def list_models(model_manager, format_filter=None):
    """List cached models."""
    print("📋 Cached Models")
    print("=" * 50)
    
    cache_info = model_manager.get_cache_info()
    
    if not cache_info['models']:
        print("No cached models found.")
        return
    
    for model_name, model_info in cache_info['models'].items():
        print(f"\n🔹 {model_name}")
        print(f"   Size: {model_info['total_size_mb']:.2f} MB")
        print(f"   Last Modified: {model_info['last_modified']}")
        
        formats_to_show = model_info['formats']
        if format_filter:
            formats_to_show = {k: v for k, v in formats_to_show.items() 
                             if k == format_filter}
        
        if formats_to_show:
            print("   Formats:")
            for format_name, format_info in formats_to_show.items():
                print(f"     {format_name.upper()}: {format_info['size_mb']:.2f} MB")
        elif format_filter:
            print(f"   No {format_filter.upper()} format available")

if __name__ == "__main__":
    main() 