"""
Cube Evaluation Module

Contains reusable functions for evaluating Rubik's cube preprocessing combinations.
Extracted from main.py to allow reuse by testing and metrics tools.

Usage:
    from cube_evaluation import (
        evaluate_cube_result,
        evaluate_preprocessing_combination,
        find_best_preprocessing_combination,
        FACE_NAMES, COLOR_TO_LETTER, EXPECTED_CENTERS
    )
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

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


def evaluate_preprocessing_combination(face_images, facelets_by_face, segmenter, classifier,
                                        preprocessor, seg_method, cc_method, force_centers=False,
                                        cc_facelets_cache=None):
    """
    Evaluate a single preprocessing combination for all faces.

    Args:
        face_images: Dict of face_name -> original image
        facelets_by_face: Dict of face_name -> facelets array (3,3,64,64,3)
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        preprocessor: ImagePreprocessor instance
        seg_method: Segmentation preprocessing method name (or None)
        cc_method: Color classification preprocessing method name (or None)
        force_centers: If True, require centers to match expected colors
        cc_facelets_cache: Optional cache of preprocessed facelets {cc_method -> {face_name -> facelets}}

    Returns:
        tuple: (cube_data, confidence_scores, is_valid, total_confidence, details)
    """
    cube_data = {}
    confidence_scores = {}
    total_confidence = 0.0

    # Check if we have cached preprocessed facelets for this seg_method + cc_method combo
    # Cache structure: {seg_method -> {cc_method -> {face_name -> preprocessed_facelets}}}
    use_cached_cc = (cc_facelets_cache is not None and
                     seg_method in cc_facelets_cache and
                     cc_method in cc_facelets_cache[seg_method])

    for face_name in FACE_NAMES:
        if face_name not in facelets_by_face:
            continue

        facelets = facelets_by_face[face_name]
        face_colors = []
        face_confidences = []

        # Get cached preprocessed facelets if available
        if use_cached_cc and face_name in cc_facelets_cache[seg_method][cc_method]:
            cached_facelets = cc_facelets_cache[seg_method][cc_method][face_name]
        else:
            cached_facelets = None

        for row in range(3):
            for col in range(3):
                if cached_facelets is not None:
                    # Use cached preprocessed facelet
                    facelet = cached_facelets[row, col]
                else:
                    facelet = facelets[row, col]
                    # Apply CC preprocessing if specified
                    if cc_method and cc_method.lower() != 'none':
                        facelet = preprocessor.apply(cc_method, facelet)

                color, conf = classifier.classify_facelet(facelet)
                face_colors.append(COLOR_TO_LETTER.get(color, '?'))
                face_confidences.append(conf)
                total_confidence += conf

        cube_data[face_name] = face_colors
        confidence_scores[face_name] = face_confidences

    is_valid, _, details = evaluate_cube_result(cube_data, force_centers)

    return cube_data, confidence_scores, is_valid, total_confidence, details


def _segment_face_with_preprocess(args):
    """Helper function for threaded segmentation preprocessing."""
    seg_m, face_name, image, preprocessor, segmenter = args

    # Apply segmentation preprocessing
    if seg_m and seg_m.lower() != 'none':
        processed_image = preprocessor.apply(seg_m, image)
    else:
        processed_image = image

    # Segment the face
    facelets = segmenter.segment(processed_image)
    return seg_m, face_name, facelets


def _preprocess_facelets_for_cc(args):
    """Helper function for threaded CC facelet preprocessing."""
    cc_method, face_name, facelets, preprocessor = args

    # Apply preprocessing to each facelet
    processed = np.empty_like(facelets)
    for row in range(3):
        for col in range(3):
            if cc_method and cc_method.lower() != 'none':
                processed[row, col] = preprocessor.apply(cc_method, facelets[row, col])
            else:
                processed[row, col] = facelets[row, col]

    return cc_method, face_name, processed


def _evaluate_cc_combination(args):
    """Helper function for threaded CC preprocessing evaluation."""
    (seg_m, cc_m, face_images, facelets_by_face, segmenter, classifier,
     preprocessor, force_centers, cc_facelets_cache) = args

    cube_data, conf_scores, is_valid, total_conf, details = evaluate_preprocessing_combination(
        face_images, facelets_by_face, segmenter, classifier,
        preprocessor, seg_m, cc_m, force_centers, cc_facelets_cache
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
                                          verbose=True, record_metrics=True):
    """
    Find the best preprocessing combination that produces valid cube results.

    Uses threading to parallelize preprocessing operations for better performance.

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

    Returns:
        tuple: (best_cube_data, best_confidences, seg_method, cc_method, all_results)
            or (None, None, None, None, all_results) if no valid combination found
    """
    methods = preprocessor.get_available_methods()

    # Determine which methods to try
    seg_methods = methods if all_seg_preprocess else [seg_method or 'none']
    cc_methods = methods if all_cc_preprocess else [cc_method or 'none']

    total_combos = len(seg_methods) * len(cc_methods)
    use_threading = total_combos > 1  # Only use threading for multiple combinations

    if verbose:
        print(f"\nEvaluating {total_combos} preprocessing combinations", end='', flush=True)

    # Progress dot printer - runs in background thread
    import threading
    stop_dots = threading.Event()

    def print_dots():
        while not stop_dots.is_set():
            if stop_dots.wait(2.0):  # Wait 2 seconds or until stopped
                break
            if verbose:
                print(".", end='', flush=True)

    dot_thread = threading.Thread(target=print_dots, daemon=True)
    dot_thread.start()

    all_results = []
    valid_results = []

    seg_time_start = time.time()

    # STAGE 1: Segmentation preprocessing (parallel by seg_method x face)
    # For each seg_method, we need to process all 6 faces
    segmented_facelets = {}  # {seg_method -> {face_name -> facelets}}

    if use_threading:
        # Build all segmentation tasks
        seg_tasks = []
        for seg_m in seg_methods:
            for face_name, image in face_images.items():
                seg_tasks.append((seg_m, face_name, image, preprocessor, segmenter))

        # Execute segmentation in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_segment_face_with_preprocess, task) for task in seg_tasks]

            for future in as_completed(futures):
                seg_m, face_name, facelets = future.result()
                if seg_m not in segmented_facelets:
                    segmented_facelets[seg_m] = {}
                segmented_facelets[seg_m][face_name] = facelets
    else:
        # Single combination - no threading
        seg_m = seg_methods[0]
        segmented_facelets[seg_m] = {}
        for face_name, image in face_images.items():
            _, _, facelets = _segment_face_with_preprocess(
                (seg_m, face_name, image, preprocessor, segmenter)
            )
            segmented_facelets[seg_m][face_name] = facelets

    seg_time = time.time() - seg_time_start

    # STAGE 2: Precompute all CC preprocessed facelets (parallel)
    cc_time_start = time.time()

    # Structure: {seg_method -> {cc_method -> {face_name -> preprocessed_facelets}}}
    cc_facelets_cache = {}

    if use_threading and len(cc_methods) > 1:
        # Build all CC preprocessing tasks
        cc_tasks = []
        for seg_m, facelets_by_face in segmented_facelets.items():
            for cc_m in cc_methods:
                for face_name, facelets in facelets_by_face.items():
                    cc_tasks.append((cc_m, face_name, facelets, preprocessor, seg_m))

        # Execute CC preprocessing in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(lambda t: (t[4], *_preprocess_facelets_for_cc(t[:4])), task)
                       for task in cc_tasks]

            for future in as_completed(futures):
                seg_m, cc_m, face_name, processed_facelets = future.result()
                if seg_m not in cc_facelets_cache:
                    cc_facelets_cache[seg_m] = {}
                if cc_m not in cc_facelets_cache[seg_m]:
                    cc_facelets_cache[seg_m][cc_m] = {}
                cc_facelets_cache[seg_m][cc_m][face_name] = processed_facelets

    # STAGE 3: Evaluate all combinations (parallel)
    eval_tasks = []
    for seg_m in seg_methods:
        facelets_by_face = segmented_facelets[seg_m]
        for cc_m in cc_methods:
            eval_tasks.append((seg_m, cc_m, face_images, facelets_by_face,
                              segmenter, classifier, preprocessor, force_centers, cc_facelets_cache))

    if use_threading:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_evaluate_cc_combination, task) for task in eval_tasks]

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                if result['is_valid']:
                    valid_results.append(result)
    else:
        # Single combination
        result = _evaluate_cc_combination(eval_tasks[0])
        all_results.append(result)
        if result['is_valid']:
            valid_results.append(result)

    cc_time = time.time() - cc_time_start

    # Stop the dot printer and finish the line
    stop_dots.set()
    dot_thread.join(timeout=0.1)
    if verbose:
        print(" done", flush=True)

    # Record metrics for ALL evaluated combinations (not just the winner)
    if record_metrics and all_results:
        from PreprocessorMetrics import get_metrics
        metrics = get_metrics()
        metrics.record_all_combinations(all_results, segmenter_name=segmenter_name)

    if not valid_results:
        if verbose:
            print("  No valid preprocessing combinations found!")
            # Show the best invalid result
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
    import os

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
                               verbose=True, record_metrics=True):
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
        record_metrics=record_metrics
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
