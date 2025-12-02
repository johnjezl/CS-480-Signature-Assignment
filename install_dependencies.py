#!/usr/bin/env python3
"""
Rubik's Cube Solver - Dependency Installation Script

This script installs all required dependencies for the Rubik's Cube Solver project.
It automatically detects the platform and installs the appropriate packages.

Usage:
    python install_dependencies.py [--training] [--jetson]

Options:
    --training    Install additional packages needed for model training
    --jetson      Skip PyTorch installation (already included in JetPack)
"""

import subprocess
import sys
import platform
import argparse
import os


def run_pip(packages, description=""):
    """Run pip install for a list of packages."""
    if description:
        print(f"\n{'=' * 50}")
        print(f"  {description}")
        print("=" * 50)

    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  [OK] {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to install {package}: {e}")
            return False
    return True


def is_jetson():
    """Detect if running on Jetson platform."""
    # Check for Jetson-specific indicators
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            if 'jetson' in model or 'tegra' in model:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    try:
        if os.path.exists('/etc/nv_tegra_release'):
            return True
    except:
        pass

    return False


def get_torch_install_command():
    """Get the appropriate PyTorch installation command for the platform."""
    system = platform.system()

    if system == "Windows":
        # Windows with CUDA 11.8 (most common)
        return [
            "torch",
            "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    elif system == "Darwin":
        # macOS
        return ["torch", "torchvision"]
    else:
        # Linux with CUDA 11.8
        return [
            "torch",
            "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]


def install_pytorch(skip_jetson=False):
    """Install PyTorch based on platform."""
    if skip_jetson or is_jetson():
        print("\n" + "=" * 50)
        print("  Jetson detected - Skipping PyTorch installation")
        print("  (PyTorch is pre-installed with JetPack)")
        print("=" * 50)
        return True

    print("\n" + "=" * 50)
    print("  Installing PyTorch")
    print("=" * 50)

    cmd = get_torch_install_command()
    print(f"\nRunning: pip install {' '.join(cmd)}")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + cmd)
        print("\n  [OK] PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] Failed to install PyTorch: {e}")
        print("\nPlease install PyTorch manually from: https://pytorch.org/get-started/locally/")
        return False


def verify_installation():
    """Verify that all required packages are installed."""
    print("\n" + "=" * 50)
    print("  Verifying Installation")
    print("=" * 50)

    packages_to_check = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
    ]

    all_ok = True
    for import_name, display_name in packages_to_check:
        try:
            __import__(import_name)
            print(f"  [OK] {display_name}")
        except ImportError:
            print(f"  [MISSING] {display_name}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Install dependencies for Rubik's Cube Solver"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Install additional packages for model training"
    )
    parser.add_argument(
        "--jetson",
        action="store_true",
        help="Skip PyTorch installation (for Jetson with JetPack)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  RUBIK'S CUBE SOLVER - DEPENDENCY INSTALLER")
    print("=" * 50)
    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")

    # Check if running on Jetson
    jetson_detected = is_jetson()
    if jetson_detected:
        print("Jetson platform detected!")
        args.jetson = True

    # Upgrade pip first
    print("\n" + "=" * 50)
    print("  Upgrading pip")
    print("=" * 50)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Core dependencies
    core_packages = [
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
    ]
    if not run_pip(core_packages, "Installing Core Dependencies"):
        print("\nCore dependency installation failed!")
        sys.exit(1)

    # PyTorch
    if not install_pytorch(skip_jetson=args.jetson):
        print("\nPyTorch installation failed!")
        print("Continuing with other packages...")

    # Training dependencies (optional)
    if args.training:
        training_packages = [
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "tqdm>=4.60.0",
        ]
        if not run_pip(training_packages, "Installing Training Dependencies"):
            print("\nTraining dependency installation had errors")

    # Verify installation
    if verify_installation():
        print("\n" + "=" * 50)
        print("  INSTALLATION COMPLETE!")
        print("=" * 50)
        print("\nAll required packages are installed.")
        print("\nYou can now run:")
        print("  python main.py")
    else:
        print("\n" + "=" * 50)
        print("  INSTALLATION INCOMPLETE")
        print("=" * 50)
        print("\nSome packages are missing. Please install them manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
