#!/usr/bin/env python3
"""
Simple launcher for the Depth Map Flask Application
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

def main():
    """Launch the Flask application."""
    print("🎯 Depth Map Project - Flask App Launcher")
    print("=" * 50)
    
    # Set PYTHONPATH
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    os.environ["PYTHONPATH"] = str(src_path)
    sys.path.insert(0, str(src_path))
    
    # Check dependencies
    try:
        import flask
        from depthmap.classical.postprocessing import DepthPostProcessor
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False
    
    print("\n🚀 Starting Flask application...")
    print("📍 The app will be available at: http://localhost:5000")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the Flask app
    try:
        from app import app
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Flask application stopped")
        return True
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        return False

if __name__ == "__main__":
    main() 