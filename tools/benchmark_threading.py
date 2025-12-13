#!/usr/bin/env python3
"""
Benchmark Sequential vs Threaded Preprocessing

Compares execution time for preprocessing operations when run
sequentially vs with ThreadPoolExecutor.

Usage:
    python tools/benchmark_threading.py
    python tools/benchmark_threading.py --iterations 5
"""

import sys
import os
import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ImagePreprocessor import ImagePreprocessor

# Try to import GPU preprocessor
try:
    from GPUImagePreprocessor import GPUImagePreprocessor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUImagePreprocessor not available, using CPU only")


def create_test_face_images(num_faces=6, size=(480, 640)):
    """Create simulated face images."""
    images = {}
    face_names = ['up', 'down', 'front', 'back', 'left', 'right']
    for i in range(num_faces):
        images[face_names[i]] = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
    return images


def run_sequential(face_images, preprocessor, methods):
    """Run preprocessing sequentially."""
    results = {}
    method_times = {}

    for method in methods:
        method_start = time.perf_counter()
        results[method] = {}
        for face_name, image in face_images.items():
            processed = preprocessor.apply(method, image)
            results[method][face_name] = processed
        method_times[method] = (time.perf_counter() - method_start) * 1000

    return results, method_times


def _process_task(args):
    """Helper for threaded processing."""
    method, face_name, image, preprocessor = args
    processed = preprocessor.apply(method, image)
    return method, face_name, processed


def run_threaded(face_images, preprocessor, methods, max_workers=None):
    """Run preprocessing with threading."""
    results = {method: {} for method in methods}
    method_times = {method: 0 for method in methods}

    # Create all tasks
    tasks = []
    task_method_map = {}
    for method in methods:
        for face_name, image in face_images.items():
            task_id = len(tasks)
            tasks.append((method, face_name, image, preprocessor))
            task_method_map[task_id] = method

    # Track per-method timing
    method_start_times = {}
    method_end_times = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_task, task): i for i, task in enumerate(tasks)}

        for future in as_completed(futures):
            task_id = futures[future]
            method, face_name, processed = future.result()
            results[method][face_name] = processed

            # Track completion time for this method's last task
            method_end_times[method] = time.perf_counter()
            if method not in method_start_times:
                method_start_times[method] = time.perf_counter()

    return results, method_times


def benchmark(face_images, preprocessor, iterations=3):
    """Run full benchmark comparing sequential vs threaded."""
    methods = preprocessor.get_available_methods()
    num_faces = len(face_images)
    num_methods = len(methods)
    total_ops = num_faces * num_methods

    print(f"\nBenchmark Configuration:")
    print(f"  Faces: {num_faces}")
    print(f"  Preprocessing methods: {num_methods}")
    print(f"  Total operations: {total_ops}")
    print(f"  Iterations: {iterations}")
    print(f"  Image size: {list(face_images.values())[0].shape[1]}x{list(face_images.values())[0].shape[0]}")

    # Warmup
    print("\nWarming up...")
    for method in methods[:3]:
        for face_name, image in list(face_images.items())[:2]:
            _ = preprocessor.apply(method, image)

    # Sequential benchmark
    print("\n" + "=" * 70)
    print("SEQUENTIAL EXECUTION")
    print("=" * 70)

    seq_times = []
    seq_method_times_all = []

    for i in range(iterations):
        start = time.perf_counter()
        _, method_times = run_sequential(face_images, preprocessor, methods)
        elapsed = (time.perf_counter() - start) * 1000
        seq_times.append(elapsed)
        seq_method_times_all.append(method_times)
        print(f"  Iteration {i+1}: {elapsed:.2f} ms")

    seq_avg = sum(seq_times) / len(seq_times)
    print(f"\n  Average: {seq_avg:.2f} ms")

    # Calculate average per-method times
    seq_method_avg = {}
    for method in methods:
        times = [mt[method] for mt in seq_method_times_all]
        seq_method_avg[method] = sum(times) / len(times)

    # Threaded benchmark
    print("\n" + "=" * 70)
    print("THREADED EXECUTION")
    print("=" * 70)

    thread_times = []

    for i in range(iterations):
        start = time.perf_counter()
        _, _ = run_threaded(face_images, preprocessor, methods)
        elapsed = (time.perf_counter() - start) * 1000
        thread_times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f} ms")

    thread_avg = sum(thread_times) / len(thread_times)
    print(f"\n  Average: {thread_avg:.2f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    speedup = seq_avg / thread_avg
    time_saved = seq_avg - thread_avg

    print(f"\n  Sequential average:  {seq_avg:.2f} ms")
    print(f"  Threaded average:    {thread_avg:.2f} ms")
    print(f"  Speedup:             {speedup:.2f}x")
    print(f"  Time saved:          {time_saved:.2f} ms ({time_saved/seq_avg*100:.1f}%)")

    # Per-method breakdown (sequential times)
    print("\n" + "-" * 70)
    print("PER-METHOD TIMES (Sequential, averaged)")
    print("-" * 70)
    print(f"{'Method':<25} {'Time (ms)':<12} {'Per Face (ms)':<15} {'% of Total'}")
    print("-" * 70)

    sorted_methods = sorted(seq_method_avg.items(), key=lambda x: x[1], reverse=True)
    for method, avg_time in sorted_methods:
        per_face = avg_time / num_faces
        pct = (avg_time / seq_avg) * 100
        print(f"{method:<25} {avg_time:<12.2f} {per_face:<15.2f} {pct:.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<25} {seq_avg:<12.2f}")

    return {
        'sequential_avg_ms': seq_avg,
        'threaded_avg_ms': thread_avg,
        'speedup': speedup,
        'method_times': seq_method_avg
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sequential vs threaded preprocessing"
    )
    parser.add_argument('--iterations', '-n', type=int, default=3,
                        help="Number of benchmark iterations (default: 3)")
    parser.add_argument('--gpu', action='store_true',
                        help="Use GPU preprocessing if available")
    parser.add_argument('--size', type=str, default='640x480',
                        help="Image size WxH (default: 640x480)")

    args = parser.parse_args()

    # Parse image size
    try:
        w, h = map(int, args.size.split('x'))
    except ValueError:
        print(f"Invalid size format: {args.size}. Use WxH (e.g., 640x480)")
        return 1

    # Select preprocessor
    if args.gpu and GPU_AVAILABLE:
        preprocessor = GPUImagePreprocessor(use_gpu=True)
        if preprocessor.is_gpu_enabled():
            print("Using GPU preprocessing (VPI/CUDA)")
        else:
            print("GPU not available, falling back to CPU")
            preprocessor = ImagePreprocessor()
    else:
        preprocessor = ImagePreprocessor()
        print("Using CPU preprocessing (OpenCV)")

    # Create test images
    print(f"\nCreating test images ({w}x{h})...")
    face_images = create_test_face_images(num_faces=6, size=(h, w))

    # Run benchmark
    results = benchmark(face_images, preprocessor, iterations=args.iterations)

    return 0


if __name__ == '__main__':
    sys.exit(main())
