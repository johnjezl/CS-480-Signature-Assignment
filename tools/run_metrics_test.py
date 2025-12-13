#!/usr/bin/env python3
"""
Comprehensive Metrics Test Script

Runs all combinations of segmenters and preprocessing methods on all available
image sets to generate comprehensive metrics data.

Usage:
    python tools/run_metrics_test.py
    python tools/run_metrics_test.py --image-set "Black Background"
    python tools/run_metrics_test.py --segmenter contour-neighbor
    python tools/run_metrics_test.py --clear-metrics  # Clear before running
    python tools/run_metrics_test.py --dry-run        # Show what would be run

This script tests:
- All 6 segmenter algorithms: grid-division, contour-perspective, contour-neighbor,
                              canny-square, brightness-otsu, auto
- All 26 preprocessing methods for both segmentation and color classification
- All 4 image sets: root input_faces, Black Background, Camera Captures, Grey_Background

Results are recorded to preprocessor_metrics.json and can be analyzed with:
    python tools/analyze_preprocessor_metrics.py
"""

import sys
import os
import time
import argparse
from datetime import datetime

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


def run_test_for_image_set(image_set_path, image_set_name, segmenter_names, classifier,
                           preprocessor, use_gpu=True, verbose=True):
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

    for seg_name in segmenter_names:
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"  Testing segmenter: {seg_name}")
            print('-' * 80)

        try:
            segmenter = Segmenter.create(seg_name)
        except Exception as e:
            print(f"  ERROR: Could not create segmenter '{seg_name}': {e}")
            continue

        start_time = time.time()

        # Run all preprocessing combinations
        best_result, all_results = evaluate_all_combinations(
            face_images, segmenter, classifier, preprocessor,
            segmenter_name=seg_name,
            force_centers=False,
            verbose=verbose,
            record_metrics=True
        )

        elapsed = time.time() - start_time

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
            'elapsed_seconds': elapsed
        }

        results_summary['total_combinations_tested'] += total_count
        results_summary['total_valid_combinations'] += valid_count

        if verbose:
            print(f"  Completed in {elapsed:.1f}s: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
            if best_result:
                print(f"  Best: seg={best_result['seg_method']}, cc={best_result['cc_method']}")

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
            print(f"    {seg_name:<25} {valid:>4}/{total:<4} valid ({rate:>5.1f}%)")

            total_combos += total
            total_valid += valid

    print(f"\n  TOTAL: {total_valid}/{total_combos} valid combinations")
    if total_combos > 0:
        print(f"  Overall success rate: {total_valid/total_combos*100:.1f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive preprocessing metrics tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run all tests
  %(prog)s --image-set black_background # Test only Black Background
  %(prog)s --segmenter contour-neighbor # Test only contour-neighbor segmenter
  %(prog)s --clear-metrics              # Clear metrics before running
  %(prog)s --dry-run                    # Show what would be run without running
  %(prog)s --no-gpu                     # Force CPU preprocessor
"""
    )

    parser.add_argument('--image-set', '-i', type=str, action='append',
                        help='Image set key to test (can specify multiple). '
                             'Options: root, black_background, camera_captures, grey_background')

    parser.add_argument('--segmenter', '-s', type=str, action='append',
                        help='Segmenter to test (can specify multiple). '
                             'Options: grid-division, contour-perspective, contour-neighbor, '
                             'canny-square, brightness-otsu, auto')

    parser.add_argument('--clear-metrics', '-c', action='store_true',
                        help='Clear existing metrics before running tests')

    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be run without actually running tests')

    parser.add_argument('--no-gpu', action='store_true',
                        help='Force using CPU preprocessor even if GPU is available')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "COMPREHENSIVE PREPROCESSING METRICS TEST".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

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
    all_segmenters = Segmenter.get_available_segmenters()
    print(f"\nAvailable segmenters: {', '.join(all_segmenters)}")

    if args.segmenter:
        segmenters_to_test = []
        for seg in args.segmenter:
            if Segmenter.is_valid(seg):
                segmenters_to_test.append(seg)
            else:
                print(f"  WARNING: Segmenter '{seg}' not found, skipping")
    else:
        segmenters_to_test = all_segmenters

    # Determine preprocessor
    use_gpu = GPU_AVAILABLE and not args.no_gpu
    if use_gpu:
        preprocessor = GPUImagePreprocessor()
        print("\nUsing GPU preprocessor")
    else:
        preprocessor = ImagePreprocessor()
        print("\nUsing CPU preprocessor")

    num_methods = len(preprocessor.get_available_methods())
    total_combos_per_segmenter = num_methods * num_methods  # seg x cc

    print(f"Preprocessing methods: {num_methods}")
    print(f"Combinations per segmenter: {total_combos_per_segmenter}")

    # Calculate total work
    total_tests = len(image_sets_to_test) * len(segmenters_to_test) * total_combos_per_segmenter
    print(f"\nTotal test combinations: {total_tests:,}")
    print(f"  ({len(image_sets_to_test)} image sets x {len(segmenters_to_test)} segmenters x {total_combos_per_segmenter} preprocessing combos)")

    if args.dry_run:
        print("\n[DRY RUN] Would run the following tests:")
        for set_key, set_info in image_sets_to_test.items():
            print(f"\n  Image Set: {set_info['name']}")
            for seg_name in segmenters_to_test:
                print(f"    - {seg_name}: {total_combos_per_segmenter} combinations")
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
    except Exception as e:
        print(f"ERROR: Could not load classifier: {e}")
        return 1

    # Run tests
    all_summaries = []
    overall_start = time.time()

    for set_key, set_info in image_sets_to_test.items():
        summary = run_test_for_image_set(
            set_info['path'],
            set_info['name'],
            segmenters_to_test,
            classifier,
            preprocessor,
            use_gpu=use_gpu,
            verbose=not args.quiet
        )
        all_summaries.append(summary)

    overall_elapsed = time.time() - overall_start

    # Print summary
    print_summary(all_summaries)

    print(f"\nTotal test time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"Metrics saved to: preprocessor_metrics.json")
    print("\nAnalyze results with:")
    print("  python tools/analyze_preprocessor_metrics.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
