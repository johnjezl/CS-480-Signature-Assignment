#!/usr/bin/env python3
"""
Metrics Test Script V2

Optimized version of run_metrics_test that:
1. Skips segmenter preprocessing for brightness-otsu (v5) since Otsu thresholding
   is adaptive and doesn't benefit from preprocessing
2. Adds memory monitoring for debugging on memory-constrained systems

Usage:
    python tools/run_metrics_test_v2.py
    python tools/run_metrics_test_v2.py --image-set black_background
    python tools/run_metrics_test_v2.py --segmenter contour-neighbor
    python tools/run_metrics_test_v2.py --clear-metrics
    python tools/run_metrics_test_v2.py --dry-run

Results are recorded to preprocessor_metrics.json and can be analyzed with:
    python tools/analyze_preprocessor_metrics.py
"""

import sys
import os
import time
import argparse
import gc
from datetime import datetime

# Early detection of --debug flag before imports (modules check DEBUG at import time)
if '--debug' in sys.argv or '-d' in sys.argv:
    os.environ['DEBUG'] = '1'

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from Segmenter import Segmenter
from FaceletColorClassifier import FaceletColorClassifier
from ImagePreprocessor import ImagePreprocessor
from PreprocessorMetrics import PreprocessorMetrics, get_metrics
from cube_evaluation import (
    evaluate_all_combinations, load_face_images, FACE_NAMES
)

# Try to import GPU preprocessor
try:
    from GPUImagePreprocessor import GPUImagePreprocessor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUImagePreprocessor = None


def get_memory_info():
    """
    Get memory usage info from /proc/meminfo.

    Returns:
        dict with keys: mem_total, mem_available, mem_used, swap_total, swap_used (all in MB)
    """
    info = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    # Convert kB to MB
                    value_kb = int(parts[1])
                    if key == 'MemTotal':
                        info['mem_total'] = value_kb / 1024
                    elif key == 'MemAvailable':
                        info['mem_available'] = value_kb / 1024
                    elif key == 'SwapTotal':
                        info['swap_total'] = value_kb / 1024
                    elif key == 'SwapFree':
                        info['swap_free'] = value_kb / 1024

        # Calculate derived values
        info['mem_used'] = info.get('mem_total', 0) - info.get('mem_available', 0)
        info['swap_used'] = info.get('swap_total', 0) - info.get('swap_free', 0)
    except Exception as e:
        info = {'mem_total': 0, 'mem_available': 0, 'mem_used': 0,
                'swap_total': 0, 'swap_used': 0, 'swap_free': 0, 'error': str(e)}

    return info


def format_memory_status(info=None):
    """Format memory status as a compact string."""
    if info is None:
        info = get_memory_info()

    if 'error' in info:
        return f"[Mem: error - {info['error']}]"

    mem_used = info.get('mem_used', 0)
    mem_total = info.get('mem_total', 1)
    mem_pct = (mem_used / mem_total) * 100 if mem_total > 0 else 0
    swap_used = info.get('swap_used', 0)

    return f"[Mem: {mem_used:.0f}/{mem_total:.0f}MB ({mem_pct:.0f}%) | Swap: {swap_used:.0f}MB]"


def print_memory_status(label="Memory"):
    """Print current memory status with a label."""
    info = get_memory_info()
    status = format_memory_status(info)
    print(f"  {label}: {status}")
    return info


# Segmenters that don't benefit from segmentation preprocessing
# brightness-otsu uses Otsu thresholding which is adaptive
NO_SEG_PREPROCESS_SEGMENTERS = {'brightness-otsu'}


# Image set directories relative to project root
IMAGE_SETS = {
    'root': 'input_faces',
    'black_background': 'input_faces/Black Background',
    'camera_captures': 'input_faces/Camera Captures',
    'grey_background': 'input_faces/Grey_Background',
}

# Friendly names for display
IMAGE_SET_NAMES = {
    'root': 'Root (Default)',
    'black_background': 'Black Background',
    'camera_captures': 'Camera Captures',
    'grey_background': 'Grey Background',
}


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def discover_image_sets():
    """Discover available image sets."""
    project_root = get_project_root()
    available = {}

    for key, rel_path in IMAGE_SETS.items():
        full_path = os.path.join(project_root, rel_path)
        if os.path.isdir(full_path):
            # Check if it has the required face images
            images = load_face_images(full_path)
            if images is not None:
                available[key] = {
                    'path': full_path,
                    'name': IMAGE_SET_NAMES.get(key, key),
                    'face_count': len(images)
                }

    return available


def format_time_remaining(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_segmenters_to_test():
    """Get list of all available segmenters."""
    return Segmenter.get_available_segmenters()


def run_test_for_image_set(image_set_path, image_set_name, segmenter_names, classifier,
                           preprocessor, use_gpu=True, verbose=True,
                           global_progress=None, max_workers=None):
    """
    Run all segmenter and preprocessing combinations for a single image set.

    Args:
        image_set_path: Path to directory containing face images
        image_set_name: Display name for the image set
        segmenter_names: List of segmenter names to test
        classifier: FaceletColorClassifier instance
        preprocessor: ImagePreprocessor instance
        use_gpu: Whether to use GPU preprocessor if available
        verbose: Print progress information
        global_progress: Dict with global progress tracking

    Returns:
        Dict with results summary
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  IMAGE SET: {image_set_name}")
        print(f"  Path: {image_set_path}")
        print('=' * 80)

    # Load face images
    face_images = load_face_images(image_set_path)
    if face_images is None:
        print(f"  ERROR: Could not load all 6 face images from {image_set_path}")
        return None

    if verbose:
        print(f"  Loaded {len(face_images)} face images")

    results_summary = {
        'image_set': image_set_name,
        'path': image_set_path,
        'segmenters': {},
        'total_combinations_tested': 0,
        'total_valid_combinations': 0,
        'start_time': datetime.now().isoformat(),
        'end_time': None
    }

    all_methods = preprocessor.get_available_methods()

    for seg_idx, seg_name in enumerate(segmenter_names):
        # Calculate and display time estimate
        if global_progress:
            completed = global_progress['completed']
            total = global_progress['total']
            elapsed_total = time.time() - global_progress['start_time']

            if completed > 0:
                avg_time = elapsed_total / completed
                remaining = (total - completed) * avg_time
                eta_str = format_time_remaining(remaining)
                pct = (completed / total) * 100
                mem_status = format_memory_status()
                print(f"\n>>> Progress: {completed}/{total} ({pct:.1f}%) | Elapsed: {format_time_remaining(elapsed_total)} | ETA: {eta_str}")
                print(f"    {mem_status}")
            else:
                mem_status = format_memory_status()
                print(f"\n>>> Progress: {completed}/{total} | Starting...")
                print(f"    {mem_status}")

        # Determine seg_methods_list based on segmenter type
        if seg_name in NO_SEG_PREPROCESS_SEGMENTERS:
            seg_methods_list = ['none']
            combo_note = "(seg preprocess skipped - Otsu is adaptive)"
        else:
            seg_methods_list = None  # Use all methods
            combo_note = ""

        if verbose:
            print(f"\n{'-' * 80}")
            print(f"  Testing segmenter: {seg_name} ({seg_idx + 1}/{len(segmenter_names)}) {combo_note}")
            print('-' * 80)

        try:
            segmenter = Segmenter.create(seg_name)
        except Exception as e:
            print(f"  ERROR: Could not create segmenter '{seg_name}': {e}")
            if global_progress:
                global_progress['completed'] += 1
            continue

        start_time = time.time()

        # Run preprocessing combinations
        best_result, all_results = evaluate_all_combinations(
            face_images, segmenter, classifier, preprocessor,
            segmenter_name=seg_name,
            force_centers=False,
            verbose=verbose,
            record_metrics=True,
            seg_methods_list=seg_methods_list,  # None = all, ['none'] = skip seg preprocess
            max_workers=max_workers
        )

        elapsed = time.time() - start_time

        # Update global progress
        if global_progress:
            global_progress['completed'] += 1
            global_progress['times'].append(elapsed)

        # Collect stats
        valid_count = sum(1 for r in all_results if r['is_valid'])
        total_count = len(all_results)

        results_summary['segmenters'][seg_name] = {
            'total_combinations': total_count,
            'valid_combinations': valid_count,
            'success_rate': (valid_count / total_count * 100) if total_count > 0 else 0,
            'best_result': {
                'seg_method': best_result['seg_method'] if best_result else None,
                'cc_method': best_result['cc_method'] if best_result else None,
            } if best_result else None,
            'elapsed_seconds': elapsed,
            'seg_preprocess_skipped': seg_name in NO_SEG_PREPROCESS_SEGMENTERS
        }

        results_summary['total_combinations_tested'] += total_count
        results_summary['total_valid_combinations'] += valid_count

        if verbose:
            print(f"  Completed in {elapsed:.1f}s: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
            if best_result:
                print(f"  Best: seg={best_result['seg_method']}, cc={best_result['cc_method']}")

        # Clean up memory after each segmenter
        gc.collect()

    results_summary['end_time'] = datetime.now().isoformat()
    return results_summary


def print_summary(all_summaries):
    """Print overall test summary."""
    print("\n" + "=" * 80)
    print("  OVERALL TEST SUMMARY")
    print("=" * 80)

    total_combos = 0
    total_valid = 0

    for summary in all_summaries:
        if summary is None:
            continue

        print(f"\n  {summary['image_set']}:")
        for seg_name, seg_results in summary['segmenters'].items():
            valid = seg_results['valid_combinations']
            total = seg_results['total_combinations']
            rate = seg_results['success_rate']
            skip_note = " (no seg preproc)" if seg_results.get('seg_preprocess_skipped') else ""
            print(f"    {seg_name:<25} {valid:>4}/{total:<4} valid ({rate:>5.1f}%){skip_note}")

            total_combos += total
            total_valid += valid

    print(f"\n  TOTAL: {total_valid}/{total_combos} valid combinations")
    if total_combos > 0:
        print(f"  Overall success rate: {total_valid/total_combos*100:.1f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run optimized preprocessing metrics tests (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimizations in V2:
  - Skips segmenter preprocessing for brightness-otsu (Otsu is adaptive)
  - Adds memory monitoring for debugging on memory-constrained systems

Examples:
  %(prog)s                              # Run all tests
  %(prog)s --image-set black_background # Test only Black Background
  %(prog)s --segmenter contour-neighbor # Test only contour-neighbor segmenter
  %(prog)s --clear-metrics              # Clear metrics before running
  %(prog)s --dry-run                    # Show what would be run without running
  %(prog)s --no-gpu                     # Force CPU preprocessor
  %(prog)s --max-workers 4              # Limit to 4 threads (reduce memory)
"""
    )

    parser.add_argument('--image-set', '-i', type=str, action='append',
                        help='Image set key to test (can specify multiple). '
                             'Options: root, black_background, camera_captures, grey_background')

    parser.add_argument('--segmenter', '-s', type=str, action='append',
                        help='Segmenter to test (can specify multiple)')

    parser.add_argument('--clear-metrics', '-c', action='store_true',
                        help='Clear existing metrics before running tests')

    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be run without actually running tests')

    parser.add_argument('--no-gpu', action='store_true',
                        help='Force using CPU preprocessor even if GPU is available')

    parser.add_argument('--max-workers', '-w', type=int, default=None,
                        help='Max threads for parallel processing. '
                             'Default: min(32, cpu_count+4). Lower values reduce memory usage.')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug logging output')

    args = parser.parse_args()

    # Enable debug mode if requested
    if args.debug:
        os.environ['DEBUG'] = '1'
        print("[DEBUG] Debug mode enabled")

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "OPTIMIZED PREPROCESSING METRICS TEST (V2)".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Print initial memory status
    print_memory_status("Initial memory")

    # Discover available image sets
    available_sets = discover_image_sets()

    if not available_sets:
        print("\nERROR: No valid image sets found!")
        return 1

    print(f"\nAvailable image sets: {len(available_sets)}")
    for key, info in available_sets.items():
        print(f"  - {key}: {info['name']} ({info['face_count']} faces)")

    # Determine which image sets to test
    if args.image_set:
        image_sets_to_test = {}
        for key in args.image_set:
            if key in available_sets:
                image_sets_to_test[key] = available_sets[key]
            else:
                print(f"  WARNING: Image set '{key}' not found, skipping")
    else:
        image_sets_to_test = available_sets

    # Determine which segmenters to test
    default_segmenters = get_segmenters_to_test()
    print(f"\nSegmenters to test: {', '.join(default_segmenters)}")
    print(f"  Note: brightness-otsu will skip segmenter preprocessing (Otsu is adaptive)")

    if args.segmenter:
        segmenters_to_test = []
        for seg in args.segmenter:
            if Segmenter.is_valid(seg):
                segmenters_to_test.append(seg)
            else:
                print(f"  WARNING: Segmenter '{seg}' not found, skipping")
    else:
        segmenters_to_test = default_segmenters

    # Determine preprocessor
    use_gpu = GPU_AVAILABLE and not args.no_gpu
    if use_gpu:
        preprocessor = GPUImagePreprocessor()
        print("\nUsing GPU preprocessor")
    else:
        preprocessor = ImagePreprocessor()
        print("\nUsing CPU preprocessor")

    # Show thread count
    if args.max_workers:
        print(f"Max worker threads: {args.max_workers}")
    else:
        import os as _os
        default_workers = min(32, (_os.cpu_count() or 1) + 4)
        print(f"Max worker threads: {default_workers} (default)")

    num_methods = len(preprocessor.get_available_methods())

    # Calculate combinations (accounting for brightness-otsu optimization)
    total_combos = 0
    combo_breakdown = []
    for seg in segmenters_to_test:
        if seg in NO_SEG_PREPROCESS_SEGMENTERS:
            combos = 1 * num_methods  # Only 'none' for seg, all for cc
            combo_breakdown.append(f"{seg}: 1 x {num_methods} = {combos}")
        else:
            combos = num_methods * num_methods
            combo_breakdown.append(f"{seg}: {num_methods} x {num_methods} = {combos}")
        total_combos += combos

    print(f"\nPreprocessing methods: {num_methods}")
    print(f"Combinations per segmenter:")
    for breakdown in combo_breakdown:
        print(f"  {breakdown}")

    # Calculate total work
    total_tests = len(image_sets_to_test) * total_combos
    print(f"\nTotal test combinations: {total_tests:,}")
    print(f"  ({len(image_sets_to_test)} image sets x {total_combos} total segmenter combos)")

    if args.dry_run:
        print("\n[DRY RUN] Would run the following tests:")
        for set_key, set_info in image_sets_to_test.items():
            print(f"\n  Image Set: {set_info['name']}")
            for seg_name in segmenters_to_test:
                if seg_name in NO_SEG_PREPROCESS_SEGMENTERS:
                    combos = num_methods
                    note = "(seg preprocess skipped)"
                else:
                    combos = num_methods * num_methods
                    note = ""
                print(f"    - {seg_name}: {combos} combinations {note}")
        print("\n[DRY RUN] No tests were actually run.")
        return 0

    # Clear metrics if requested
    if args.clear_metrics:
        print("\nClearing existing metrics...")
        metrics = get_metrics()
        metrics.clear_data()

    # Load classifier once
    print("\nLoading color classifier...")
    try:
        classifier = FaceletColorClassifier()
        print_memory_status("After classifier load")
    except Exception as e:
        print(f"ERROR: Could not load classifier: {e}")
        return 1

    # Run tests
    all_summaries = []
    overall_start = time.time()

    # Initialize global progress tracking
    total_units = len(image_sets_to_test) * len(segmenters_to_test)
    global_progress = {
        'completed': 0,
        'total': total_units,
        'start_time': overall_start,
        'times': []
    }

    print(f"\n>>> Starting test run: {total_units} segmenter evaluations total")

    for set_idx, (set_key, set_info) in enumerate(image_sets_to_test.items()):
        summary = run_test_for_image_set(
            set_info['path'],
            set_info['name'],
            segmenters_to_test,
            classifier,
            preprocessor,
            use_gpu=use_gpu,
            verbose=not args.quiet,
            global_progress=global_progress,
            max_workers=args.max_workers
        )
        all_summaries.append(summary)

        # Memory status after each image set
        print(f"\n>>> Image set {set_idx + 1}/{len(image_sets_to_test)} complete")
        print_memory_status("After image set")
        gc.collect()

    overall_elapsed = time.time() - overall_start

    # Print summary
    print_summary(all_summaries)

    print(f"\nTotal test time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"Metrics saved to: preprocessor_metrics.json")
    print_memory_status("Final memory")
    print("\nAnalyze results with:")
    print("  python tools/analyze_preprocessor_metrics.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
