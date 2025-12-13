#!/usr/bin/env python3
"""
Benchmark GPU vs CPU preprocessing performance.

Compares execution time for all preprocessing methods between
GPUImagePreprocessor (VPI/CUDA) and ImagePreprocessor (CPU/OpenCV).

Usage:
    python tools/benchmark_preprocessor.py
    python tools/benchmark_preprocessor.py --iterations 20
    python tools/benchmark_preprocessor.py --image path/to/image.jpg
"""

import sys
import os
import time
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ImagePreprocessor import ImagePreprocessor

# Try to import GPU preprocessor
try:
    from GPUImagePreprocessor import GPUImagePreprocessor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUImagePreprocessor not available")


def create_test_images():
    """Create test images of different sizes."""
    images = {
        'facelet_64x64': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        'face_640x480': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        'face_1280x720': np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8),
        'face_1920x1080': np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
    }
    return images


def benchmark_method(preprocessor, method_name, image, iterations=10, warmup=2):
    """Benchmark a single preprocessing method."""
    # Warmup runs
    for _ in range(warmup):
        try:
            _ = preprocessor.apply(method_name, image)
        except Exception:
            return None, "Error"

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            _ = preprocessor.apply(method_name, image)
        except Exception as e:
            return None, str(e)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        'avg': avg_time * 1000,  # Convert to ms
        'min': min_time * 1000,
        'max': max_time * 1000,
    }, None


def run_benchmark(image, iterations=10):
    """Run full benchmark comparing GPU vs CPU."""
    cpu_preprocessor = ImagePreprocessor()

    if GPU_AVAILABLE:
        gpu_preprocessor = GPUImagePreprocessor(use_gpu=True)
        gpu_enabled = gpu_preprocessor.is_gpu_enabled()
    else:
        gpu_preprocessor = None
        gpu_enabled = False

    methods = cpu_preprocessor.get_available_methods()

    results = []

    print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")
    print(f"Iterations: {iterations}")
    print(f"GPU available: {gpu_enabled}")
    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10} {'Winner'}")
    print("=" * 80)

    for method in methods:
        # CPU benchmark
        cpu_result, cpu_error = benchmark_method(cpu_preprocessor, method, image, iterations)

        # GPU benchmark
        if gpu_enabled:
            gpu_result, gpu_error = benchmark_method(gpu_preprocessor, method, image, iterations)
        else:
            gpu_result, gpu_error = None, "N/A"

        # Calculate speedup
        if cpu_result and gpu_result:
            speedup = cpu_result['avg'] / gpu_result['avg']
            winner = "GPU" if speedup > 1.0 else "CPU"
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup = None
            winner = "-"
            speedup_str = "-"

        cpu_str = f"{cpu_result['avg']:.2f}" if cpu_result else cpu_error
        gpu_str = f"{gpu_result['avg']:.2f}" if gpu_result else (gpu_error if gpu_error else "N/A")

        print(f"{method:<25} {cpu_str:<15} {gpu_str:<15} {speedup_str:<10} {winner}")

        results.append({
            'method': method,
            'cpu_ms': cpu_result['avg'] if cpu_result else None,
            'gpu_ms': gpu_result['avg'] if gpu_result else None,
            'speedup': speedup,
            'winner': winner
        })

    print("=" * 80)

    # Summary statistics
    gpu_wins = sum(1 for r in results if r['winner'] == 'GPU')
    cpu_wins = sum(1 for r in results if r['winner'] == 'CPU')

    valid_speedups = [r['speedup'] for r in results if r['speedup'] is not None]
    if valid_speedups:
        avg_speedup = sum(valid_speedups) / len(valid_speedups)
        max_speedup = max(valid_speedups)
        min_speedup = min(valid_speedups)
    else:
        avg_speedup = max_speedup = min_speedup = 0

    print(f"\nSummary:")
    print(f"  GPU wins: {gpu_wins}/{len(results)}")
    print(f"  CPU wins: {cpu_wins}/{len(results)}")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Max speedup (GPU): {max_speedup:.2f}x")
    print(f"  Min speedup (GPU): {min_speedup:.2f}x")

    return results


def run_multi_image_benchmark(iterations=10):
    """Run benchmarks on multiple image sizes."""
    images = create_test_images()

    all_results = {}

    for name, image in images.items():
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {name}")
        print(f"{'='*80}")
        all_results[name] = run_benchmark(image, iterations)

    # Final comparison across image sizes
    print("\n" + "=" * 80)
    print("SUMMARY BY IMAGE SIZE")
    print("=" * 80)

    print(f"\n{'Image Size':<20} {'GPU Wins':<12} {'CPU Wins':<12} {'Avg Speedup'}")
    print("-" * 60)

    for name, results in all_results.items():
        gpu_wins = sum(1 for r in results if r['winner'] == 'GPU')
        cpu_wins = sum(1 for r in results if r['winner'] == 'CPU')
        valid_speedups = [r['speedup'] for r in results if r['speedup'] is not None]
        avg_speedup = sum(valid_speedups) / len(valid_speedups) if valid_speedups else 0

        print(f"{name:<20} {gpu_wins:<12} {cpu_wins:<12} {avg_speedup:.2f}x")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs CPU preprocessing performance"
    )
    parser.add_argument('--iterations', '-n', type=int, default=10,
                        help="Number of iterations per method (default: 10)")
    parser.add_argument('--image', '-i', type=str, default=None,
                        help="Path to test image (default: use generated images)")
    parser.add_argument('--all-sizes', '-a', action='store_true',
                        help="Test all image sizes (default)")

    args = parser.parse_args()

    if args.image:
        # Load user-provided image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            return 1
        print(f"Using image: {args.image}")
        run_benchmark(image, args.iterations)
    else:
        # Run multi-image benchmark
        run_multi_image_benchmark(args.iterations)

    return 0


if __name__ == '__main__':
    sys.exit(main())
