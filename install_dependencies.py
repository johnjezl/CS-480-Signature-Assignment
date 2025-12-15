#!/usr/bin/env python3
"""
Rubik's Cube Solver - Dependency Installation Script

This script installs all required dependencies for the Rubik's Cube Solver project.
It automatically detects the platform and installs the appropriate packages.

Usage:
    python install_dependencies.py [--training] [--jetson] [--skip-pytorch]

Options:
    --training      Install additional packages needed for model training
    --jetson        Install Jetson-specific PyTorch with CUDA support
    --skip-pytorch  Skip PyTorch installation entirely
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
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
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


def get_jetpack_version():
    """Get the JetPack/L4T version from the Jetson system."""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read()
            # Parse "# R36 (release), REVISION: 4.7, ..."
            if 'R36' in content:
                return 6  # JetPack 6.x
            elif 'R35' in content:
                return 5  # JetPack 5.x
            elif 'R34' in content or 'R32' in content:
                return 4  # JetPack 4.x
    except (FileNotFoundError, PermissionError):
        pass
    return None


# Jetson PyTorch wheel URLs by JetPack version
# Updated December 2024 - check NVIDIA forums for latest:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# NOTE: JetPack 6.2+ (L4T R36.4+) may have library compatibility issues with these wheels.
#       If you encounter libcudnn or libcusparseLt errors, use CPU PyTorch instead:
#       pip3 install torch --index-url https://download.pytorch.org/whl/cpu
JETSON_PYTORCH_URLS = {
    # JetPack 6.1 (L4T R36.3)
    6: "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl",
    # JetPack 5.1.2 (L4T R35.4)
    5: "https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl",
}


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


def install_jetson_pytorch():
    """Install PyTorch with CUDA support for Jetson platforms."""
    print("\n" + "=" * 50)
    print("  Installing PyTorch for Jetson (with CUDA)")
    print("=" * 50)

    jetpack_version = get_jetpack_version()
    if jetpack_version is None:
        print("\n  [WARNING] Could not detect JetPack version")
        print("  Please install PyTorch manually from:")
        print("  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
        return False

    print(f"\n  Detected JetPack {jetpack_version}.x")

    if jetpack_version not in JETSON_PYTORCH_URLS:
        print(f"\n  [ERROR] No PyTorch wheel URL configured for JetPack {jetpack_version}")
        print("  Please check NVIDIA forums for the correct wheel:")
        print("  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
        return False

    wheel_url = JETSON_PYTORCH_URLS[jetpack_version]
    print(f"\n  Installing from: {wheel_url}")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--break-system-packages", "--no-cache", wheel_url
        ])
        print("\n  [OK] PyTorch for Jetson installed successfully")

        # Verify CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  [OK] CUDA is available! Device: {torch.cuda.get_device_name(0)}")
            else:
                print("  [WARNING] PyTorch installed but CUDA not available")
                print("  This may indicate a driver or library mismatch")
        except Exception as e:
            print(f"  [WARNING] Could not verify CUDA: {e}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] Failed to install PyTorch: {e}")
        print("\n  Please install manually:")
        print(f"  pip3 install --no-cache {wheel_url}")
        return False


def install_pytorch(for_jetson=False, skip=False):
    """Install PyTorch based on platform."""
    if skip:
        print("\n" + "=" * 50)
        print("  Skipping PyTorch installation (--skip-pytorch)")
        print("=" * 50)
        return True

    if for_jetson:
        return install_jetson_pytorch()

    print("\n" + "=" * 50)
    print("  Installing PyTorch")
    print("=" * 50)

    cmd = get_torch_install_command()
    print(f"\nRunning: pip install {' '.join(cmd)}")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages"] + cmd)
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
        help="Install Jetson-specific PyTorch with CUDA support (auto-detected)"
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch installation entirely"
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
        if not args.skip_pytorch:
            args.jetson = True
            jetpack_ver = get_jetpack_version()
            if jetpack_ver:
                print(f"JetPack version: {jetpack_ver}.x")

    # Upgrade pip first
    print("\n" + "=" * 50)
    print("  Upgrading pip")
    print("=" * 50)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "--upgrade", "pip"])

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
    if not install_pytorch(for_jetson=args.jetson, skip=args.skip_pytorch):
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
