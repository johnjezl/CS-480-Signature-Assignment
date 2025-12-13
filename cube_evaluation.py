"""
Cube Evaluation Module

Contains reusable functions for evaluating Rubik's cube preprocessing combinations.
Extracted from main.py to allow reuse by testing and metrics tools.

Performance optimizations:
- Batch CNN inference (54 facelets in one call)
- Multiprocessing for CPU-bound segmentation/preprocessing
- Early termination option
- Vectorized facelet preprocessing

Usage:
    from cube_evaluation import (
        evaluate_cube_result,
        evaluate_preprocessing_combination,
        find_best_preprocessing_combination,
        FACE_NAMES, COLOR_TO_LETTER, EXPECTED_CENTERS
    )
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time
import os

# Face names for the solver (in order of input)
FACE_NAMES = ['up', 'down', 'front', 'back', 'left', 'right']
FACE_DISPLAY_NAMES = ['Up (Yellow)', 'Down (White)', 'Front (Blue)',
                      'Back (Green)', 'Left (Orange)', 'Right (Red)']

# Color name to letter mapping
COLOR_TO_LETTER = {
    'white': 'W', 'yellow': 'Y', 'red': 'R',
    'orange': 'O', 'blue': 'B', 'green': 'G'
}

# Expected center colors for each face (standard cube orientation)
EXPECTED_CENTERS = {
    'up': 'Y',      # Yellow
    'down': 'W',    # White
    'front': 'B',   # Blue
    'back': 'G',    # Green
    'left': 'O',    # Orange
    'right': 'R',   # Red
}


def evaluate_cube_result(cube_data, force_centers=False):
    """
    Evaluate whether a cube result is valid.

    Args:
        cube_data: Dict of face_name -> list of 9 color letters
        force_centers: If True, require center facelets to match expected colors

    Returns:
        tuple: (is_valid, total_confidence, details)
    """
    color_counts = {'W': 0, 'Y': 0, 'R': 0, 'O': 0, 'B': 0, 'G': 0}
    total_confidence = 0
    centers_match = True
    center_details = {}

    for face_name, colors in cube_data.items():
        for color in colors:
            if color in color_counts:
                color_counts[color] += 1

        # Check center (index 4 in a 3x3 face)
        center_color = colors[4] if len(colors) > 4 else None
        expected_center = EXPECTED_CENTERS.get(face_name)
        center_details[face_name] = {
            'actual': center_color,
            'expected': expected_center,
            'match': center_color == expected_center
        }
        if center_color != expected_center:
            centers_match = False

    # Check if all colors appear exactly 9 times
    is_even = all(count == 9 for count in color_counts.values())

    # Validity depends on even distribution and optionally center matching
    is_valid = is_even and (not force_centers or centers_match)

    details = {
        'color_counts': color_counts,
        'is_even': is_even,
        'centers_match': centers_match,
        'center_details': center_details
    }

    return is_valid, total_confidence, details


def evaluate_preprocessing_combination_batch(facelets_by_face, classifier, force_centers=False):
    """
    Evaluate a preprocessing combination using batch inference.

    This is the optimized version that classifies all 54 facelets in one batch.

    Args:
        facelets_by_face: Dict of face_name -> facelets array (3,3,64,64,3)
        classifier: FaceletColorClassifier instance
        force_centers: If True, require centers to match expected colors

    Returns:
        tuple: (cube_data, confidence_scores, is_valid, total_confidence, details)
    """
    # Use batch inference for all faces at once
    classifications = classifier.classify_multiple_faces(facelets_by_face)

    cube_data = {}
    confidence_scores = {}
    total_confidence = 0.0

    for face_name in FACE_NAMES:
        if face_name not in classifications:
            continue

        face_classifications = classifications[face_name]
        face_colors = []
        face_confidences = []

        for row in range(3):
            for col in range(3):
                color, conf = face_classifications[row, col]
                face_colors.append(COLOR_TO_LETTER.get(color, '?'))
                face_confidences.append(conf)
                total_confidence += conf

        cube_data[face_name] = face_colors
        confidence_scores[face_name] = face_confidences

    is_valid, _, details = evaluate_cube_result(cube_data, force_centers)

    return cube_data, confidence_scores, is_valid, total_confidence, details


def _segment_face_with_preprocess(args):
    """Helper function for parallel segmentation preprocessing."""
    seg_m, face_name, image, preprocessor, segmenter = args

    # Apply segmentation preprocessing
    if seg_m and seg_m.lower() != 'none':
        processed_image = preprocessor.apply(seg_m, image)
    else:
        processed_image = image

    # Segment the face
    facelets = segmenter.segment(processed_image)
    return seg_m, face_name, facelets


def _preprocess_facelets_vectorized(facelets, preprocessor, cc_method):
    """
    Apply preprocessing to all 9 facelets of a face.

    Args:
        facelets: Array (3, 3, 64, 64, 3)
        preprocessor: ImagePreprocessor instance
        cc_method: Preprocessing method name

    Returns:
        Preprocessed facelets array (3, 3, 64, 64, 3)
    """
    if not cc_method or cc_method.lower() == 'none':
        return facelets

    processed = np.empty_like(facelets)
    for row in range(3):
        for col in range(3):
            processed[row, col] = preprocessor.apply(cc_method, facelets[row, col])

    return processed


def _preprocess_all_facelets_for_cc(seg_m, facelets_by_face, preprocessor, cc_methods):
    """
    Preprocess all facelets for all CC methods.

    Returns:
        Dict: {cc_method -> {face_name -> preprocessed_facelets}}
    """
    result = {}
    for cc_m in cc_methods:
        result[cc_m] = {}
        for face_name, facelets in facelets_by_face.items():
            result[cc_m][face_name] = _preprocess_facelets_vectorized(
                facelets, preprocessor, cc_m
            )
    return result


def _evaluate_single_combination(args):
    """
    Evaluate a single seg+cc combination.

    This function is designed to be called in a worker process.
    Note: classifier must be created fresh in each process if using multiprocessing.
    """
    seg_m, cc_m, facelets_by_face, force_centers, classifier = args

    cube_data, conf_scores, is_valid, total_conf, details = evaluate_preprocessing_combination_batch(
        facelets_by_face, classifier, force_centers
    )

    return {
        'seg_method': seg_m,
        'cc_method': cc_m,
        'cube_data': cube_data,
        'confidence_scores': conf_scores,
        'is_valid': is_valid,
        'total_confidence': total_conf,
        'details': details
    }


def find_best_preprocessing_combination(face_images, segmenter, classifier, preprocessor,
                                          all_seg_preprocess=False, all_cc_preprocess=False,
                                          seg_method=None, cc_method=None, force_centers=False,
                                          max_workers=None, segmenter_name: str = 'unknown',
                                          verbose=True, record_metrics=True,
                                          early_stop_confidence=None):
    """
    Find the best preprocessing combination that produces valid cube results.

    Uses batch inference and optional multiprocessing for better performance.

    Args:
        face_images: Dict of face_name -> original image
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        preprocessor: ImagePreprocessor instance
        all_seg_preprocess: If True, try all segmentation preprocessing methods
        all_cc_preprocess: If True, try all CC preprocessing methods
        seg_method: Single segmentation preprocessing method (if not all)
        cc_method: Single CC preprocessing method (if not all)
        force_centers: If True, require centers to match expected colors
        max_workers: Max thread pool workers (default: min(32, cpu_count + 4))
        segmenter_name: Name of the segmenter algorithm for metrics tracking
        verbose: If True, print progress information
        record_metrics: If True, record metrics to PreprocessorMetrics
        early_stop_confidence: If set, stop early when a result exceeds this threshold

    Returns:
        tuple: (best_cube_data, best_confidences, seg_method, cc_method, all_results)
            or (None, None, None, None, all_results) if no valid combination found
    """
    methods = preprocessor.get_available_methods()

    # Determine which methods to try
    seg_methods = methods if all_seg_preprocess else [seg_method or 'none']
    cc_methods = methods if all_cc_preprocess else [cc_method or 'none']

    total_combos = len(seg_methods) * len(cc_methods)
    use_parallel = total_combos > 1

    if verbose:
        print(f"\nEvaluating {total_combos} preprocessing combinations", end='', flush=True)

    # Progress indicator
    import threading
    stop_dots = threading.Event()
    combo_count = [0]  # Use list for mutable reference in closure

    def print_progress():
        while not stop_dots.is_set():
            if stop_dots.wait(2.0):
                break
            if verbose:
                print(".", end='', flush=True)

    progress_thread = threading.Thread(target=print_progress, daemon=True)
    progress_thread.start()

    all_results = []
    valid_results = []
    early_stopped = False

    # STAGE 1: Segmentation preprocessing (parallel)
    seg_time_start = time.time()
    segmented_facelets = {}  # {seg_method -> {face_name -> facelets}}

    if use_parallel and len(seg_methods) > 1:
        # Use threading for segmentation (OpenCV releases GIL)
        seg_tasks = []
        for seg_m in seg_methods:
            for face_name, image in face_images.items():
                seg_tasks.append((seg_m, face_name, image, preprocessor, segmenter))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_segment_face_with_preprocess, task) for task in seg_tasks]

            for future in as_completed(futures):
                seg_m, face_name, facelets = future.result()
                if seg_m not in segmented_facelets:
                    segmented_facelets[seg_m] = {}
                segmented_facelets[seg_m][face_name] = facelets
    else:
        # Single seg method or sequential processing
        for seg_m in seg_methods:
            segmented_facelets[seg_m] = {}
            for face_name, image in face_images.items():
                _, _, facelets = _segment_face_with_preprocess(
                    (seg_m, face_name, image, preprocessor, segmenter)
                )
                segmented_facelets[seg_m][face_name] = facelets

    seg_time = time.time() - seg_time_start

    # STAGE 2: CC preprocessing and evaluation
    cc_time_start = time.time()

    # For each seg method, preprocess all facelets for all cc methods, then evaluate
    for seg_m in seg_methods:
        if early_stopped:
            break

        facelets_by_face = segmented_facelets[seg_m]

        # Preprocess facelets for all CC methods
        if use_parallel and len(cc_methods) > 1:
            # Precompute all CC preprocessed facelets
            cc_preprocessed = _preprocess_all_facelets_for_cc(
                seg_m, facelets_by_face, preprocessor, cc_methods
            )

            # Evaluate all CC methods
            for cc_m in cc_methods:
                if early_stopped:
                    break

                preprocessed_facelets = cc_preprocessed[cc_m]

                # Use batch inference
                cube_data, conf_scores, is_valid, total_conf, details = evaluate_preprocessing_combination_batch(
                    preprocessed_facelets, classifier, force_centers
                )

                result = {
                    'seg_method': seg_m,
                    'cc_method': cc_m,
                    'cube_data': cube_data,
                    'confidence_scores': conf_scores,
                    'is_valid': is_valid,
                    'total_confidence': total_conf,
                    'details': details
                }

                all_results.append(result)
                if is_valid:
                    valid_results.append(result)

                    # Check for early stopping
                    if early_stop_confidence and total_conf >= early_stop_confidence:
                        early_stopped = True
                        if verbose:
                            print(f" [early stop at {total_conf:.0f}]", end='')
                        break

                combo_count[0] += 1
        else:
            # Single CC method
            cc_m = cc_methods[0]
            preprocessed_facelets = {
                face_name: _preprocess_facelets_vectorized(facelets, preprocessor, cc_m)
                for face_name, facelets in facelets_by_face.items()
            }

            cube_data, conf_scores, is_valid, total_conf, details = evaluate_preprocessing_combination_batch(
                preprocessed_facelets, classifier, force_centers
            )

            result = {
                'seg_method': seg_m,
                'cc_method': cc_m,
                'cube_data': cube_data,
                'confidence_scores': conf_scores,
                'is_valid': is_valid,
                'total_confidence': total_conf,
                'details': details
            }

            all_results.append(result)
            if is_valid:
                valid_results.append(result)

    cc_time = time.time() - cc_time_start

    # Stop progress indicator
    stop_dots.set()
    progress_thread.join(timeout=0.1)
    if verbose:
        status = " done" if not early_stopped else ""
        print(status, flush=True)

    # Record metrics for ALL evaluated combinations
    if record_metrics and all_results:
        from PreprocessorMetrics import get_metrics
        metrics = get_metrics()
        metrics.record_all_combinations(all_results, segmenter_name=segmenter_name)

    if not valid_results:
        if verbose:
            print("  No valid preprocessing combinations found!")
            if all_results:
                best_invalid = max(all_results, key=lambda r: r['total_confidence'])
                print(f"  Best invalid result: seg={best_invalid['seg_method']}, cc={best_invalid['cc_method']}")
        return None, None, None, None, all_results

    # Sort valid results by total confidence (descending)
    valid_results.sort(key=lambda r: r['total_confidence'], reverse=True)

    # Check for ties at the top
    top_confidence = valid_results[0]['total_confidence']
    top_results = [r for r in valid_results if abs(r['total_confidence'] - top_confidence) < 0.01]

    if len(top_results) > 1:
        # Prefer 'none' preprocessing when tied
        for r in top_results:
            if r['seg_method'] == 'none' and r['cc_method'] == 'none':
                best = r
                break
            elif r['seg_method'] == 'none':
                best = r
                break
            elif r['cc_method'] == 'none':
                best = r
                break
        else:
            best = top_results[0]
    else:
        best = valid_results[0]

    return (best['cube_data'], best['confidence_scores'],
            best['seg_method'], best['cc_method'], all_results)


def load_face_images(directory, face_names=None):
    """
    Load face images from a directory.

    Args:
        directory: Path to directory containing face images
        face_names: List of face names to look for (default: FACE_NAMES)

    Returns:
        Dict of face_name -> image (BGR numpy array), or None if not all found
    """
    import cv2

    if face_names is None:
        face_names = FACE_NAMES

    images = {}

    # Try different file patterns
    for face_name in face_names:
        found = False
        for pattern in [f"{face_name}.jpg", f"{face_name}.JPG",
                        f"{face_name.upper()}.jpg", f"{face_name.upper()}.JPG",
                        f"{face_name}.png", f"{face_name.upper()}.png",
                        f"{face_name.lower()}.jpg", f"{face_name.lower()}.png"]:
            path = os.path.join(directory, pattern)
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    images[face_name] = img
                    found = True
                    break
        if not found:
            return None

    return images


def evaluate_all_combinations(face_images, segmenter, classifier, preprocessor,
                               segmenter_name: str = 'unknown', force_centers=False,
                               verbose=True, record_metrics=True,
                               early_stop_confidence=None):
    """
    Evaluate ALL preprocessing combinations and return results.

    This is a simplified wrapper around find_best_preprocessing_combination
    that always tries all combinations.

    Args:
        face_images: Dict of face_name -> image
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        preprocessor: ImagePreprocessor instance
        segmenter_name: Name of the segmenter for metrics
        force_centers: If True, require centers to match expected colors
        verbose: If True, print progress
        record_metrics: If True, record to PreprocessorMetrics
        early_stop_confidence: If set, stop early when a result exceeds this

    Returns:
        tuple: (best_result, all_results)
            best_result is dict with best combination or None
            all_results is list of all evaluated results
    """
    best_cube_data, best_confidences, best_seg, best_cc, all_results = find_best_preprocessing_combination(
        face_images, segmenter, classifier, preprocessor,
        all_seg_preprocess=True,
        all_cc_preprocess=True,
        force_centers=force_centers,
        segmenter_name=segmenter_name,
        verbose=verbose,
        record_metrics=record_metrics,
        early_stop_confidence=early_stop_confidence
    )

    if best_cube_data is not None:
        best_result = {
            'cube_data': best_cube_data,
            'confidence_scores': best_confidences,
            'seg_method': best_seg,
            'cc_method': best_cc,
            'is_valid': True
        }
    else:
        best_result = None

    return best_result, all_results


# Legacy compatibility functions
def evaluate_preprocessing_combination(face_images, facelets_by_face, segmenter, classifier,
                                        preprocessor, seg_method, cc_method, force_centers=False,
                                        cc_facelets_cache=None):
    """
    Legacy function for evaluating a single preprocessing combination.

    For better performance, use evaluate_preprocessing_combination_batch directly.
    """
    # Apply CC preprocessing if needed
    if cc_method and cc_method.lower() != 'none':
        preprocessed = {}
        for face_name, facelets in facelets_by_face.items():
            preprocessed[face_name] = _preprocess_facelets_vectorized(
                facelets, preprocessor, cc_method
            )
    else:
        preprocessed = facelets_by_face

    return evaluate_preprocessing_combination_batch(preprocessed, classifier, force_centers)


def _preprocess_facelets_for_cc(args):
    """Legacy helper function for threaded CC facelet preprocessing."""
    cc_method, face_name, facelets, preprocessor = args
    processed = _preprocess_facelets_vectorized(facelets, preprocessor, cc_method)
    return cc_method, face_name, processed


def _evaluate_cc_combination(args):
    """Legacy helper function for threaded CC preprocessing evaluation."""
    (seg_m, cc_m, face_images, facelets_by_face, segmenter, classifier,
     preprocessor, force_centers, cc_facelets_cache) = args

    # Apply CC preprocessing
    if cc_m and cc_m.lower() != 'none':
        preprocessed = {}
        for face_name, facelets in facelets_by_face.items():
            preprocessed[face_name] = _preprocess_facelets_vectorized(
                facelets, preprocessor, cc_m
            )
    else:
        preprocessed = facelets_by_face

    cube_data, conf_scores, is_valid, total_conf, details = evaluate_preprocessing_combination_batch(
        preprocessed, classifier, force_centers
    )

    return {
        'seg_method': seg_m,
        'cc_method': cc_m,
        'cube_data': cube_data,
        'confidence_scores': conf_scores,
        'is_valid': is_valid,
        'total_confidence': total_conf,
        'details': details
    }
