#!/usr/bin/env python3
"""
Setup script for Depth Map Project in VS Code
This script sets up the project to work with VS Code and virtual environments.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description="", check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def create_vscode_settings():
    """Create VS Code settings for the project."""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Settings.json
    settings = {
        "python.defaultInterpreterPath": "./venv/Scripts/python.exe" if platform.system() == "Windows" else "./venv/bin/python",
        "python.terminal.activateEnvironment": True,
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": False,
        "python.linting.flake8Enabled": True,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length", "88"],
        "python.testing.pytestEnabled": True,
        "python.testing.pytestArgs": ["tests/"],
        "files.associations": {
            "*.yml": "yaml",
            "*.yaml": "yaml"
        },
        "python.analysis.extraPaths": ["./src"],
        "terminal.integrated.env.windows": {
            "PYTHONPATH": "${workspaceFolder}/src"
        },
        "terminal.integrated.env.linux": {
            "PYTHONPATH": "${workspaceFolder}/src"
        },
        "terminal.integrated.env.osx": {
            "PYTHONPATH": "${workspaceFolder}/src"
        }
    }
    
    settings_file = vscode_dir / "settings.json"
    import json
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
    
    print("✅ VS Code settings created")

def create_launch_config():
    """Create VS Code launch configuration."""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Flask App",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/app.py",
                "console": "integratedTerminal",
                "env": {
                    "FLASK_ENV": "development",
                    "FLASK_DEBUG": "1",
                    "PYTHONPATH": "${workspaceFolder}/src"
                }
            },
            {
                "name": "Demo Script",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/demo.py",
                "console": "integratedTerminal",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}/src"
                }
            },
            {
                "name": "Test Run",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/test_run.py",
                "console": "integratedTerminal",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}/src"
                }
            }
        ]
    }
    
    launch_file = vscode_dir / "launch.json"
    import json
    with open(launch_file, 'w') as f:
        json.dump(launch_config, f, indent=4)
    
    print("✅ VS Code launch configuration created")

def create_tasks_config():
    """Create VS Code tasks configuration."""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    tasks_config = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Install Dependencies",
                "type": "shell",
                "command": "pip",
                "args": ["install", "-r", "requirements.txt"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "panel": "new"
                }
            },
            {
                "label": "Run Flask App",
                "type": "shell",
                "command": "python",
                "args": ["app.py"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "panel": "new"
                },
                "options": {
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                }
            },
            {
                "label": "Run Tests",
                "type": "shell",
                "command": "python",
                "args": ["-m", "pytest", "tests/", "-v"],
                "group": "test",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "panel": "new"
                }
            },
            {
                "label": "Run Demo",
                "type": "shell",
                "command": "python",
                "args": ["demo.py"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "panel": "new"
                },
                "options": {
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                }
            }
        ]
    }
    
    tasks_file = vscode_dir / "tasks.json"
    import json
    with open(tasks_file, 'w') as f:
        json.dump(tasks_config, f, indent=4)
    
    print("✅ VS Code tasks configuration created")

def main():
    """Main setup function."""
    print("🎯 Depth Map Project - VS Code Setup")
    print("=" * 50)
    print("Setting up the project for VS Code with virtual environment...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Create virtual environment
    venv_path = "venv"
    if not os.path.exists(venv_path):
        if not run_command(f"{sys.executable} -m venv {venv_path}", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command
    if platform.system() == "Windows":
        activate_cmd = f"{venv_path}\\Scripts\\activate && "
        pip_cmd = f"{venv_path}\\Scripts\\pip"
        python_cmd = f"{venv_path}\\Scripts\\python"
    else:
        activate_cmd = f"source {venv_path}/bin/activate && "
        pip_cmd = f"{venv_path}/bin/pip"
        python_cmd = f"{venv_path}/bin/python"
    
    # Upgrade pip
    if not run_command(f"{python_cmd} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️ Pip upgrade failed, continuing...")
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Install the package in development mode
    if not run_command(f"{pip_cmd} install -e .", "Installing project in development mode"):
        return False
    
    # Create VS Code configuration
    create_vscode_settings()
    create_launch_config()
    create_tasks_config()
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("✅ Project directories created")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Open this folder in VS Code")
    print("2. Select the Python interpreter from the virtual environment:")
    print(f"   - Windows: .\\{venv_path}\\Scripts\\python.exe")
    print(f"   - Linux/Mac: ./{venv_path}/bin/python")
    print("3. Run the Flask app:")
    print("   - Press F5 to run 'Flask App' configuration, or")
    print("   - Use terminal: python app.py")
    print("4. Open http://localhost:5000 in your browser")
    print("\n🛠️ Available VS Code tasks (Ctrl+Shift+P > Tasks: Run Task):")
    print("   - Install Dependencies")
    print("   - Run Flask App")
    print("   - Run Tests")
    print("   - Run Demo")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 