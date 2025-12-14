"""
Rubik's Cube Face Scanner - Main Application

Menu-driven application with multiple modes:
1. Single Face Mode: Scan one image file and classify colors
2. Full Cube Mode: Scan all 6 faces from files and solve the cube
3. Camera Single Face: Capture one face from camera and classify (Jetson only)
4. Camera Full Cube: Capture all 6 faces from camera and solve (Jetson only)

Usage:
    python main.py [--display] [--segmenter NAME] [--rotate] [--no-animation]
                   [--segmenter-preprocess METHOD] [--cc-preprocess METHOD]
                   [--all-segmenter-preprocess] [--all-cc-preprocess] [--force-centers]
                   [--adaptive]

Options:
    --display    Show captured images on display (for Jetson with monitor)
    --segmenter NAME    Segmentation algorithm to use (default: auto)
    --rotate     Rotate camera images 180 degrees (for inverted camera mounting)
    --no-animation Disable solution animation (on by default with --display)
    --segmenter-preprocess METHOD   Preprocess image before segmentation
    --cc-preprocess METHOD          Preprocess image before color classification
    --all-segmenter-preprocess      Try all preprocessing methods for segmentation
    --all-cc-preprocess             Try all preprocessing methods for color classification
    --force-centers                 Force center facelets to match expected colors
    --adaptive                      Use adaptive evaluation with two-result confirmation

Segmentation Algorithms:
    auto               - Auto-select best algorithm based on image analysis (default)
    brightness-otsu    - Brightness-based detection with Otsu thresholding
    canny-square       - Canny edge detection with square finding
    contour-neighbor   - Contour-based detection with neighbor validation
    contour-perspective - Contour detection with perspective correction
    grid-division      - Basic grid division (original algorithm)

Preprocessing Methods:
    none, bilateral, bilateral-strong, clahe-lab, clahe-hsv, unsharp, histeq,
    histeq-v, morph-open, morph-close, satboost, satboost-mild, white-balance,
    gamma-bright, gamma-dark, median, gaussian, contrast-stretch, bilateral-clahe,
    bilateral-sat, clahe-sat, full-pipeline
"""

import cv2
import numpy as np
import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from Segmenter import Segmenter
from FaceletColorClassifier import FaceletColorClassifier
from IDASolver import IDASolver, KociembaSolver
from ImagePreprocessor import ImagePreprocessor
from CubeRenderer import CubeRenderer, CubeState
from CubeOrientationCorrector import CubeOrientationCorrector
from PreprocessorMetrics import PreprocessorMetrics, get_metrics
from DisplayManager import DisplayManager, get_platform
from cube_evaluation import (
    evaluate_cube_result, evaluate_preprocessing_combination,
    FACE_NAMES, FACE_DISPLAY_NAMES, COLOR_TO_LETTER, EXPECTED_CENTERS,
    _segment_face_with_preprocess, _preprocess_facelets_for_cc, _evaluate_cc_combination,
    load_face_images
)
from adaptive_evaluator import AdaptiveEvaluator

# Try to import GPU preprocessor (requires VPI on Jetson)
try:
    from GPUImagePreprocessor import GPUImagePreprocessor
    GPU_PREPROCESSOR_AVAILABLE = True
except ImportError:
    GPU_PREPROCESSOR_AVAILABLE = False
    GPUImagePreprocessor = None

# Try to import Jetson camera module
try:
    from JetsonCamera import JetsonCamera, is_jetson, debug_print, suppress_output
    JETSON_AVAILABLE = is_jetson()
except ImportError:
    JETSON_AVAILABLE = False
    JetsonCamera = None

    def is_jetson():
        return False

    def debug_print(msg):
        pass

    from contextlib import contextmanager
    @contextmanager
    def suppress_output():
        yield

# Global display manager instance (initialized in main())
_display_manager: DisplayManager = None


def get_display_manager() -> DisplayManager:
    """Get the global display manager instance."""
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayManager()
    return _display_manager


# Color abbreviation map
COLOR_ABBREV = {
    'white': 'W', 'yellow': 'Y', 'red': 'R',
    'orange': 'O', 'blue': 'B', 'green': 'G'
}

# Note: FACE_NAMES, FACE_DISPLAY_NAMES, COLOR_TO_LETTER, EXPECTED_CENTERS
# are now imported from cube_evaluation module

# ANSI color codes for terminal cube visualization
class TermColors:
    RESET = '\033[0m'
    # Background colors
    BG_WHITE = '\033[107m'
    BG_YELLOW = '\033[103m'
    BG_RED = '\033[101m'
    BG_ORANGE = '\033[48;5;208m'
    BG_BLUE = '\033[104m'
    BG_GREEN = '\033[102m'
    BG_BLACK = '\033[40m'
    # Foreground colors
    FG_BLACK = '\033[30m'
    FG_WHITE = '\033[97m'

    @classmethod
    def get_color(cls, color_letter):
        """Get background and foreground color codes for a color letter."""
        color_map = {
            'W': (cls.BG_WHITE, cls.FG_BLACK),
            'Y': (cls.BG_YELLOW, cls.FG_BLACK),
            'R': (cls.BG_RED, cls.FG_WHITE),
            'O': (cls.BG_ORANGE, cls.FG_BLACK),
            'B': (cls.BG_BLUE, cls.FG_WHITE),
            'G': (cls.BG_GREEN, cls.FG_BLACK),
        }
        return color_map.get(color_letter.upper(), (cls.BG_BLACK, cls.FG_WHITE))


def animate_solution(cube_data, moves_list, delay_ms=30, frames_per_move=20):
    """
    Animate the solution moves on a 3D cube visualization.

    Args:
        cube_data: Dict with face colors {'up': ['W','W',...], 'down': [...], ...}
        moves_list: List of move strings ['R', "U'", 'F2', ...]
        delay_ms: Delay between animation frames in milliseconds
        frames_per_move: Number of frames per move animation
    """
    import math
    import select
    import sys

    dm = get_display_manager()

    # Map cube_data face names to CubeState face names
    face_map = {'up': 'U', 'down': 'D', 'front': 'F', 'back': 'B', 'right': 'R', 'left': 'L'}
    color_map = {'W': 'W', 'Y': 'Y', 'R': 'R', 'O': 'O', 'B': 'B', 'G': 'G'}

    # Create cube state from scanned data
    cube = CubeState()
    for face_name, face_key in face_map.items():
        if face_name in cube_data:
            colors = cube_data[face_name]
            for i in range(9):
                row, col = i // 3, i % 3
                cube.faces[face_key][row, col] = color_map.get(colors[i], 'W')

    # Use a reasonable window size
    window_width, window_height = 800, 800

    renderer = CubeRenderer(window_width, window_height)
    window_name = "Solution Animation"

    # Create window - use AUTOSIZE for better compatibility across platforms
    dm.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Flush any pending key events from previous operations
    for _ in range(10):
        dm.waitKey(1)

    print(f"\nAnimating {len(moves_list)} moves...")
    print("In terminal: press Enter to pause/resume, 'q' to skip")

    paused = False
    move_idx = 0

    while move_idx < len(moves_list):
        move = moves_list[move_idx]

        # Animate this move using while loop so pause can freeze frame
        frame = 0
        while frame <= frames_per_move:
            angle_fraction = frame / frames_per_move

            # Ease in-out for smoother animation
            ease_fraction = 0.5 - 0.5 * math.cos(math.pi * angle_fraction)

            img = renderer.render_frame(cube, move, ease_fraction)

            # Add move counter to image (this is the only place move info is shown)
            status_text = "[PAUSED] " if paused else ""
            cv2.putText(img, f"{status_text}Move {move_idx + 1}/{len(moves_list)}: {move}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)

            dm.imshow(window_name, img)

            # Brief wait to control animation speed
            dm.waitKey(delay_ms if not paused else 50)

            # Check for terminal input (more reliable than cv2.waitKey on Jetson)
            try:
                if select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char == 'q' or char == 'Q':
                        print("Animation skipped.")
                        dm.destroyAllWindows()
                        dm.waitKey(1)
                        return
                    elif char == ' ' or char == '\n':
                        paused = not paused
                        if paused:
                            print("Paused. Press space/enter to resume, 'q' to skip.")
                        else:
                            print("Resumed.")
            except (TypeError, ValueError, OSError):
                pass

            # Only advance frame if not paused
            if not paused:
                frame += 1

        # Apply the move to cube state
        cube.apply_move(move)
        move_idx += 1

    # Show final solved state
    print("Animation complete!")
    final_img = renderer.render_frame(cube, None, 0)
    cv2.putText(final_img, "SOLVED!", (window_width // 2 - 100, window_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.putText(final_img, "Press any key to continue",
                (window_width // 2 - 200, window_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    dm.imshow(window_name, final_img)

    # Wait for any key press (with timeout to stay responsive)
    timeout_count = 0
    max_timeout = 300  # 30 seconds max wait (300 * 100ms)
    while timeout_count < max_timeout:
        key = dm.waitKey(100)
        if key != -1:
            break
        # Also check for Enter key in terminal (Unix only)
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
        except (TypeError, ValueError, OSError):
            # select doesn't work with stdin on Windows
            pass
        timeout_count += 1

    # Properly clean up OpenCV windows
    dm.destroyAllWindows()
    dm.waitKey(1)



# Note: The following are now imported from cube_evaluation module:
# - FACE_NAMES, FACE_DISPLAY_NAMES, COLOR_TO_LETTER, EXPECTED_CENTERS
# - evaluate_cube_result, evaluate_preprocessing_combination
# - find_best_preprocessing_combination
# - _segment_face_with_preprocess, _preprocess_facelets_for_cc, _evaluate_cc_combination


def find_best_preprocessing_combination(face_images, segmenter, classifier, preprocessor,
                                          all_seg_preprocess=False, all_cc_preprocess=False,
                                          seg_method=None, cc_method=None, force_centers=False,
                                          max_workers=None, segmenter_name: str = 'unknown'):
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

    # Print user-facing message with dot progress indicator
    print(f"\nEvaluating {total_combos} preprocessing combinations", end='', flush=True)

    # Debug info
    debug_print(f"Segmenter methods: {len(seg_methods)}")
    debug_print(f"Classifier methods: {len(cc_methods)}")
    debug_print(f"Total combinations: {total_combos}")
    if use_threading:
        debug_print("Using threaded execution")

    # Progress dot printer - runs in background thread
    import threading
    stop_dots = threading.Event()
    def print_dots():
        while not stop_dots.is_set():
            if stop_dots.wait(2.0):  # Wait 2 seconds or until stopped
                break
            print(".", end='', flush=True)

    dot_thread = threading.Thread(target=print_dots, daemon=True)
    dot_thread.start()

    # First, segment all faces with each segmentation preprocessing method
    # Cache segmented facelets to avoid re-segmenting for each CC method
    segmented_cache = {}  # seg_method -> {face_name -> facelets}

    start_time = time.time()

    if use_threading and len(seg_methods) > 1:
        # Threaded segmentation preprocessing
        seg_tasks = []
        for seg_m in seg_methods:
            segmented_cache[seg_m] = {}
            for face_name, image in face_images.items():
                seg_tasks.append((seg_m, face_name, image, preprocessor, segmenter))

        debug_print(f"Segmenting {len(seg_tasks)} face/method combinations...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_segment_face_with_preprocess, task) for task in seg_tasks]
            completed = 0
            for future in as_completed(futures):
                seg_m, face_name, facelets = future.result()
                if facelets is not None:
                    segmented_cache[seg_m][face_name] = facelets
                completed += 1
    else:
        # Sequential segmentation
        for seg_m in seg_methods:
            segmented_cache[seg_m] = {}
            for face_name, image in face_images.items():
                if seg_m and seg_m.lower() != 'none':
                    processed_image = preprocessor.apply(seg_m, image)
                else:
                    processed_image = image
                facelets = segmenter.segment(processed_image)
                if facelets is not None:
                    segmented_cache[seg_m][face_name] = facelets

    seg_time = time.time() - start_time
    debug_print(f"Segmentation completed in {seg_time:.2f}s")

    # Build CC facelets cache - preprocess all facelets once per (seg_method, cc_method) pair
    # This reduces calls from 26,136 (484 combos * 54 facelets) to at most 22*22*6*9 = 26,136
    # but only 22*6*9 = 1,188 actual preprocessing operations (since same seg produces same facelets)
    # Cache structure: {seg_method -> {cc_method -> {face_name -> preprocessed_facelets}}}
    cc_facelets_cache = {}

    cache_start = time.time()

    if len(cc_methods) > 1:
        debug_print(f"Building CC facelets cache for {len(seg_methods)} seg x {len(cc_methods)} cc methods...")

        # For each segmentation method's facelets, preprocess for all CC methods
        cc_preprocess_tasks = []

        for seg_m in seg_methods:
            if seg_m not in segmented_cache or not segmented_cache[seg_m]:
                continue
            cc_facelets_cache[seg_m] = {}
            for cc_m in cc_methods:
                cc_facelets_cache[seg_m][cc_m] = {}
                for face_name, facelets in segmented_cache[seg_m].items():
                    cc_preprocess_tasks.append((seg_m, cc_m, face_name, facelets, preprocessor))

        if cc_preprocess_tasks:
            debug_print(f"Preprocessing {len(cc_preprocess_tasks)} CC facelet tasks...")

            if use_threading:
                # Threaded CC facelet preprocessing
                def _preprocess_cc_task(args):
                    seg_m, cc_m, face_name, facelets, preprocessor = args
                    processed = np.empty_like(facelets)
                    for row in range(3):
                        for col in range(3):
                            if cc_m and cc_m.lower() != 'none':
                                processed[row, col] = preprocessor.apply(cc_m, facelets[row, col])
                            else:
                                processed[row, col] = facelets[row, col]
                    return seg_m, cc_m, face_name, processed

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_preprocess_cc_task, task)
                               for task in cc_preprocess_tasks]
                    for future in as_completed(futures):
                        seg_m, cc_m, face_name, processed = future.result()
                        cc_facelets_cache[seg_m][cc_m][face_name] = processed
            else:
                # Sequential CC facelet preprocessing
                for seg_m, cc_m, face_name, facelets, _ in cc_preprocess_tasks:
                    processed = np.empty_like(facelets)
                    for row in range(3):
                        for col in range(3):
                            if cc_m and cc_m.lower() != 'none':
                                processed[row, col] = preprocessor.apply(cc_m, facelets[row, col])
                            else:
                                processed[row, col] = facelets[row, col]
                    cc_facelets_cache[seg_m][cc_m][face_name] = processed

            cache_time = time.time() - cache_start
            debug_print(f"CC cache built in {cache_time:.2f}s")

    # Now try all CC preprocessing combinations
    valid_results = []
    all_results = []

    start_time = time.time()

    if use_threading and len(cc_methods) > 1:
        # Threaded CC preprocessing evaluation
        cc_tasks = []
        for seg_m in seg_methods:
            if seg_m not in segmented_cache or not segmented_cache[seg_m]:
                continue
            for cc_m in cc_methods:
                cc_tasks.append((seg_m, cc_m, face_images, segmented_cache[seg_m],
                                segmenter, classifier, preprocessor, force_centers, cc_facelets_cache))

        debug_print(f"Evaluating {len(cc_tasks)} CC combinations...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_evaluate_cc_combination, task) for task in cc_tasks]
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                if result['is_valid']:
                    valid_results.append(result)
                completed += 1
    else:
        # Sequential CC evaluation
        combo_num = 0
        for seg_m in seg_methods:
            if seg_m not in segmented_cache or not segmented_cache[seg_m]:
                continue
            for cc_m in cc_methods:
                combo_num += 1

                cube_data, conf_scores, is_valid, total_conf, details = evaluate_preprocessing_combination(
                    face_images, segmented_cache[seg_m], segmenter, classifier,
                    preprocessor, seg_m, cc_m, force_centers, cc_facelets_cache
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

    cc_time = time.time() - start_time

    # Stop the dot printer and finish the line
    stop_dots.set()
    dot_thread.join(timeout=0.1)
    print(" done", flush=True)

    debug_print(f"CC evaluation completed in {cc_time:.2f}s")
    debug_print(f"Total time: {seg_time + cc_time:.2f}s")

    # Record metrics for ALL evaluated combinations (not just the winner)
    if all_results:
        metrics = get_metrics()
        metrics.record_all_combinations(all_results, segmenter_name=segmenter_name)
        debug_print(f"Recorded metrics for {len(all_results)} combinations (segmenter={segmenter_name})")

    if not valid_results:
        print("  No valid preprocessing combinations found!")
        # Show the best invalid result
        if all_results:
            best_invalid = max(all_results, key=lambda r: r['total_confidence'])
            print(f"  Best invalid result: seg={best_invalid['seg_method']}, cc={best_invalid['cc_method']}")
            debug_print(f"Color counts: {best_invalid['details']['color_counts']}")
        return None, None, None, None, all_results

    # Sort valid results by total confidence (descending)
    valid_results.sort(key=lambda r: r['total_confidence'], reverse=True)

    # Check for ties at the top
    top_confidence = valid_results[0]['total_confidence']
    top_results = [r for r in valid_results if abs(r['total_confidence'] - top_confidence) < 0.01]

    if len(top_results) > 1:
        debug_print(f"WARNING: {len(top_results)} combinations tied with confidence {top_confidence:.1f}")
        for r in top_results[:5]:  # Show up to 5
            debug_print(f"  seg={r['seg_method']}, cc={r['cc_method']}")

        # If there's a true tie, raise error
        if len(top_results) > 1:
            # Check if they produce the same cube_data
            first_data = str(top_results[0]['cube_data'])
            all_same = all(str(r['cube_data']) == first_data for r in top_results)
            if all_same:
                debug_print("All tied results produce the same cube state - using first.")
            else:
                print("  ERROR: Tied results produce different cube states!")
                return None, None, None, None, all_results

    best = valid_results[0]
    print(f"  Best: seg={best['seg_method']}, cc={best['cc_method']} (confidence: {best['total_confidence']:.1f})")

    return (best['cube_data'], best['confidence_scores'],
            best['seg_method'], best['cc_method'], all_results)


def print_cube_net(cube_data, use_color=True):
    """
    Print the cube as an unfolded net in the terminal.

    Layout:
              +-------+
              |  UP   |
              +-------+
    +-------+-------+-------+-------+
    | LEFT  | FRONT | RIGHT | BACK  |
    +-------+-------+-------+-------+
              +-------+
              | DOWN  |
              +-------+

    Args:
        cube_data: Dict with keys 'up', 'down', 'front', 'back', 'left', 'right'
                   each containing a list of 9 color letters (W, Y, R, O, B, G)
        use_color: Whether to use ANSI colors
    """
    cell_width = 5
    face_width = cell_width * 3 + 4  # 3 cells + separators

    def h_border(count=1):
        single = "+" + "-" * cell_width
        return (single * 3 + "+") * count

    def empty_face():
        return " " * face_width

    def render_face_row(face_colors, row):
        """Render one row (3 cells) of a face."""
        start_idx = row * 3
        cells = []
        for i in range(3):
            letter = face_colors[start_idx + i]
            cell_content = f"  {letter}  "
            if use_color:
                bg, fg = TermColors.get_color(letter)
                cells.append(f"{bg}{fg}{cell_content}{TermColors.RESET}")
            else:
                cells.append(cell_content)
        return "|" + "|".join(cells) + "|"

    # Print header
    print("\n" + "=" * 50)
    print("  CUBE VISUALIZATION")
    print("=" * 50)

    # UP face (centered above)
    indent = empty_face()
    print(indent + h_border())
    print(indent + f"|{'UP':^{face_width-2}}|")
    print(indent + h_border())

    up_colors = cube_data.get('up', ['?'] * 9)
    for row in range(3):
        print(indent + render_face_row(up_colors, row))
    print(indent + h_border())

    print()

    # Middle row: LEFT, FRONT, RIGHT, BACK
    middle_faces = [('left', 'LEFT'), ('front', 'FRONT'), ('right', 'RIGHT'), ('back', 'BACK')]

    # Headers
    print(h_border(4))
    label_row = ""
    for _, label in middle_faces:
        label_row += f"|{label:^{face_width-2}}|"
    print(label_row)
    print(h_border(4))

    # Content rows
    for row in range(3):
        content_row = ""
        for face_key, _ in middle_faces:
            face_colors = cube_data.get(face_key, ['?'] * 9)
            content_row += render_face_row(face_colors, row)
        print(content_row)

    print(h_border(4))

    print()

    # DOWN face (centered below)
    print(indent + h_border())
    print(indent + f"|{'DOWN':^{face_width-2}}|")
    print(indent + h_border())

    down_colors = cube_data.get('down', ['?'] * 9)
    for row in range(3):
        print(indent + render_face_row(down_colors, row))
    print(indent + h_border())

    # Count colors across all faces
    color_counts = {'W': 0, 'Y': 0, 'R': 0, 'O': 0, 'B': 0, 'G': 0}
    for face_key in ['up', 'down', 'front', 'back', 'left', 'right']:
        for color_letter in cube_data.get(face_key, []):
            if color_letter in color_counts:
                color_counts[color_letter] += 1

    # Display color distribution
    color_names = {'W': 'White', 'Y': 'Yellow', 'R': 'Red', 'O': 'Orange', 'B': 'Blue', 'G': 'Green'}
    print("\nColor Distribution (expected: 9 each):")
    counts_valid = True
    for letter in ['W', 'Y', 'R', 'O', 'B', 'G']:
        count = color_counts[letter]
        bar = '#' * count
        status = "" if count == 9 else f" ({'missing ' + str(9 - count) if count < 9 else 'extra ' + str(count - 9)})"
        if count != 9:
            counts_valid = False
        if use_color:
            bg, fg = TermColors.get_color(letter)
            color_block = f"{bg}{fg} {letter} {TermColors.RESET}"
        else:
            color_block = f"[{letter}]"
        print(f"  {color_block} {color_names[letter]:8s}: {count:2d} {bar}{status}")
    print()

    return counts_valid


def print_classification_results(classifications, face_name=None):
    """
    Print the classification results in a formatted grid with ANSI colors.

    Args:
        classifications: 3x3 numpy array of (color, confidence) tuples
        face_name: Optional name of the face being classified

    Returns:
        face_string: String of 9 color letters (e.g., "YYYYYYYY")
    """
    if face_name:
        print(f"\n{'=' * 50}")
        print(f"CLASSIFICATION RESULTS - {face_name}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("CLASSIFICATION RESULTS")
        print("=" * 50)

    # Helper to format a colored cell
    def format_cell(color_name, conf):
        letter = COLOR_ABBREV.get(color_name, '?')
        bg, fg = TermColors.get_color(letter)
        return f"{bg}{fg}  {letter}  {TermColors.RESET}"

    # Print visual grid with colors
    cell_width = 5
    h_border = ("+" + "-" * cell_width) * 3 + "+"

    print(f"\nFace Layout:")
    print(h_border)
    for row in range(3):
        row_str = "|"
        for col in range(3):
            color, conf = classifications[row, col]
            row_str += format_cell(color, conf) + "|"
        print(row_str)
        print(h_border)

    # Print detailed results with confidence
    print("\nDetailed Results:")
    for row in range(3):
        for col in range(3):
            color, conf = classifications[row, col]
            letter = COLOR_ABBREV.get(color, '?')
            bg, fg = TermColors.get_color(letter)
            color_block = f"{bg}{fg} {letter} {TermColors.RESET}"
            print(f"  [{row},{col}]: {color_block} {color:8s} ({conf:5.1f}%)")

    # Generate face string for solver
    face_string = ""
    face_array = []
    for row in range(3):
        for col in range(3):
            color, _ = classifications[row, col]
            face_string += COLOR_ABBREV[color]
            face_array.append(COLOR_ABBREV[color])

    print(f"\nFace String: {face_string}")
    print("=" * 50)

    return face_string, face_array


def get_image_path(prompt="Enter the path to a Rubik's Cube face image (jpg or png):"):
    """
    Prompt user for an image path and validate it.

    Args:
        prompt: The prompt message to display

    Returns:
        image_path if valid, None if user wants to cancel
    """
    print(f"\n{prompt}")
    print("(Enter 'q' to cancel)")
    image_path = input("> ").strip()

    # Check for cancel
    if image_path.lower() == 'q':
        return None

    # Remove quotes if present
    if image_path.startswith('"') and image_path.endswith('"'):
        image_path = image_path[1:-1]
    if image_path.startswith("'") and image_path.endswith("'"):
        image_path = image_path[1:-1]

    # Validate file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return None

    # Validate file extension
    valid_extensions = ['.jpg', '.jpeg', '.png']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        print(f"Error: Unsupported file format '{ext}'")
        print(f"Supported formats: {', '.join(valid_extensions)}")
        return None

    return image_path


def process_single_face(image_path, segmenter, classifier, side_name=None, display=False,
                        segmenter_preprocess=None, cc_preprocess=None, preprocessor=None):
    """
    Process a single face image from file: segment and classify.

    Args:
        image_path: Path to the image file
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        side_name: Name of the cube side for logging
        display: If True, show face and facelets on display
        segmenter_preprocess: Preprocessing method name for segmentation
        cc_preprocess: Preprocessing method name for color classification
        preprocessor: ImagePreprocessor instance

    Returns:
        tuple: (classifications, facelets) or (None, None) on error
    """
    # Load image
    debug_print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None, None
    debug_print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    return process_image(image, segmenter, classifier, side_name, display,
                        segmenter_preprocess, cc_preprocess, preprocessor)


def process_image(image, segmenter, classifier, side_name=None, display=False,
                  segmenter_preprocess=None, cc_preprocess=None, preprocessor=None):
    """
    Process an image (BGR numpy array): segment and classify.

    Args:
        image: BGR numpy array (height, width, 3)
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        side_name: Name of the cube side for logging (up, down, front, back, left, right)
        display: If True, show face and facelets on display
        segmenter_preprocess: Preprocessing method name for segmentation (or None)
        cc_preprocess: Preprocessing method name for color classification (or None)
        preprocessor: ImagePreprocessor instance (required if preprocessing is specified)

    Returns:
        tuple: (classifications, facelets) or (None, None) on error
    """
    debug_print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Preprocess for segmentation (start from original image)
    if segmenter_preprocess and preprocessor and segmenter_preprocess.lower() != 'none':
        debug_print(f"Preprocessing for segmentation: {segmenter_preprocess}")
        seg_image = preprocessor.apply(segmenter_preprocess, image)
    else:
        seg_image = image

    # Segment the image into facelets
    debug_print("Segmenting image into 3x3 facelets...")
    start_time = time.time()
    facelets = segmenter.segment(seg_image)
    segment_time = time.time() - start_time
    debug_print(f"Facelets shape: {facelets.shape} (took {segment_time:.3f}s)")

    # Display face and facelets if requested
    if display:
        window_name = f"Segmented: {side_name}" if side_name else "Segmented Face"
        dm = get_display_manager()
        dm.display_face_and_facelets(seg_image, facelets, window_name)

    # Preprocess for color classification (start from original image)
    # Apply preprocessing to each facelet independently
    if cc_preprocess and preprocessor and cc_preprocess.lower() != 'none':
        debug_print(f"Preprocessing facelets for color classification: {cc_preprocess}")
        preprocessed_facelets = np.zeros_like(facelets)
        for row in range(3):
            for col in range(3):
                preprocessed_facelets[row, col] = preprocessor.apply(cc_preprocess, facelets[row, col])
        facelets_for_classification = preprocessed_facelets
    else:
        facelets_for_classification = facelets

    # Classify the colors
    debug_print("Classifying facelet colors...")
    start_time = time.time()
    classifications = classifier.classify_face(facelets_for_classification)
    classify_time = time.time() - start_time
    debug_print(f"Classification complete (took {classify_time:.3f}s)")

    return classifications, facelets


def single_face_mode(segmenter_name: str = 'auto',
                     segmenter_preprocess: str = None, cc_preprocess: str = None,
                     use_gpu: bool = True):
    """Mode 1: Process a single face image."""
    print("\n" + "=" * 50)
    print("  SINGLE FACE MODE")
    print("=" * 50)

    # Initialize segmenter
    debug_print(f"Initializing segmenter '{segmenter_name}'...")
    start_time = time.time()
    with suppress_output():
        segmenter = Segmenter.create(segmenter_name)
    segmenter_time = time.time() - start_time
    debug_print(f"Segmenter ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor if needed
    preprocessor = None
    if segmenter_preprocess or cc_preprocess:
        debug_print(f"Initializing GPUImagePreprocessor (GPU={'enabled' if use_gpu else 'disabled'})...")
        preprocessor = GPUImagePreprocessor(use_gpu=use_gpu)
        if segmenter_preprocess:
            debug_print(f"  Segmenter preprocessing: {segmenter_preprocess}")
        if cc_preprocess:
            debug_print(f"  Color classifier preprocessing: {cc_preprocess}")

    # Get image path
    image_path = get_image_path()
    if image_path is None:
        print("Cancelled.")
        return

    # Process the face
    classifications, facelets = process_single_face(image_path, segmenter, classifier, side_name="single",
                                          segmenter_preprocess=segmenter_preprocess,
                                          cc_preprocess=cc_preprocess, preprocessor=preprocessor)
    if classifications is None:
        return

    # Output results
    face_string, _ = print_classification_results(classifications)
    print(f"\nScan complete! Face string: {face_string}")


def get_directory_path():
    """
    Prompt user for a directory path and validate it.

    Returns:
        directory_path if valid, None if user wants to cancel
    """
    print("\nEnter the path to a directory containing cube face images:")
    print("(Expected files: up, down, front, back, left, right with .jpg or .png extension)")
    print("(Enter 'q' to cancel)")
    dir_path = input("> ").strip()

    # Check for cancel
    if dir_path.lower() == 'q':
        return None

    # Remove quotes if present
    if dir_path.startswith('"') and dir_path.endswith('"'):
        dir_path = dir_path[1:-1]
    if dir_path.startswith("'") and dir_path.endswith("'"):
        dir_path = dir_path[1:-1]

    # Validate directory exists
    if not os.path.exists(dir_path):
        print(f"Error: Directory not found: {dir_path}")
        return None

    if not os.path.isdir(dir_path):
        print(f"Error: Path is not a directory: {dir_path}")
        return None

    return dir_path


def find_face_images(directory):
    """
    Find face images in a directory by looking for files named
    up, down, front, back, left, right (case-insensitive).

    Args:
        directory: Path to the directory to search

    Returns:
        Dictionary mapping face_key to file path, or None if not all faces found
    """
    valid_extensions = ['.jpg', '.jpeg', '.png']
    face_files = {}

    # Get all files in directory
    try:
        files = os.listdir(directory)
    except OSError as e:
        print(f"Error reading directory: {e}")
        return None

    # Look for each face
    for face_key in FACE_NAMES:
        found = False
        for filename in files:
            name, ext = os.path.splitext(filename)
            if name.lower() == face_key and ext.lower() in valid_extensions:
                face_files[face_key] = os.path.join(directory, filename)
                found = True
                break

        if not found:
            print(f"Error: Could not find image for '{face_key}' face")
            print(f"  Expected: {face_key}.jpg, {face_key}.png, or {face_key}.jpeg")

    # Check if all faces were found
    if len(face_files) != 6:
        print(f"\nFound {len(face_files)}/6 face images.")
        return None

    return face_files


def full_cube_mode(segmenter_name: str = 'auto', display: bool = False,
                   segmenter_preprocess: str = None, cc_preprocess: str = None,
                   animate: bool = False, all_seg_preprocess: bool = False,
                   all_cc_preprocess: bool = False, force_centers: bool = False,
                   use_gpu: bool = True, adaptive: bool = False):
    """Mode 2: Process all 6 faces and solve the cube."""
    print("\n" + "=" * 50)
    print("  FULL CUBE SOLVER MODE")
    print("=" * 50)
    print("\nProvide a directory containing 6 face images:")
    print("  Required files: up, down, front, back, left, right")
    print("  Supported formats: .jpg, .jpeg, .png")
    print("  (filenames are case-insensitive)")

    # Get directory path
    dir_path = get_directory_path()
    if dir_path is None:
        print("Cancelled.")
        return

    # Find face images
    print(f"\nSearching for face images in: {dir_path}")
    face_files = find_face_images(dir_path)
    if face_files is None:
        print("\nAborting cube solve.")
        return

    # Show found files
    print("\nFound all 6 face images:")
    for face_key in FACE_NAMES:
        print(f"  {face_key}: {os.path.basename(face_files[face_key])}")

    # Initialize segmenter
    debug_print(f"Initializing segmenter '{segmenter_name}'...")
    start_time = time.time()
    with suppress_output():
        segmenter = Segmenter.create(segmenter_name)
    segmenter_time = time.time() - start_time
    debug_print(f"Segmenter ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor (always needed for all_* modes)
    debug_print(f"Initializing GPUImagePreprocessor (GPU={'enabled' if use_gpu else 'disabled'})...")
    preprocessor = GPUImagePreprocessor(use_gpu=use_gpu)
    if segmenter_preprocess:
        debug_print(f"  Segmenter preprocessing: {segmenter_preprocess}")
    if cc_preprocess:
        debug_print(f"  Color classifier preprocessing: {cc_preprocess}")

    # Dictionary to store face data for the solver
    cube_data = {}
    captured_images = {}
    captured_facelets = []

    # Check if we need to do multi-preprocessing evaluation
    use_multi_preprocess = all_seg_preprocess or all_cc_preprocess
    use_adaptive = adaptive and not use_multi_preprocess

    if use_adaptive:
        # Adaptive mode: use historical metrics to select best preprocessing
        print("\nLoading all face images...")
        for face_key in FACE_NAMES:
            image_path = face_files[face_key]
            image = cv2.imread(image_path)
            if image is None:
                print(f"\nError: Could not load image for {face_key}")
                return
            captured_images[face_key] = image
            print(f"  Loaded {face_key}: {os.path.basename(image_path)}")

        # Use adaptive evaluation
        print("\nRunning adaptive evaluation (two-result confirmation)...")
        evaluator = AdaptiveEvaluator()
        result = evaluator.evaluate(
            captured_images, classifier, preprocessor,
            max_attempts=100,
            combos_per_round=5,
            background_samples=10,
            force_centers=force_centers,
            record_metrics=True,
            verbose=True
        )

        if not result['is_valid']:
            if result.get('valid_results_count', 0) > 0:
                print(f"\nWARNING: Found {result['valid_results_count']} valid result(s) but could not confirm.")
                print("No two results matched - cube state may be ambiguous.")
            else:
                print("\nError: No valid result found. Aborting cube solve.")
            return

        cube_data = result['cube_data']

        # Convert captured_images dict to list for display
        captured_images_list = [captured_images[face_key] for face_key in FACE_NAMES]

    elif use_multi_preprocess:
        # Load all face images first
        print("\nLoading all face images...")
        for face_key in FACE_NAMES:
            image_path = face_files[face_key]
            image = cv2.imread(image_path)
            if image is None:
                print(f"\nError: Could not load image for {face_key}")
                return
            captured_images[face_key] = image
            print(f"  Loaded {face_key}: {os.path.basename(image_path)}")

        # Use the multi-preprocessing evaluation function
        best_cube_data, best_confidences, best_seg, best_cc, all_results = find_best_preprocessing_combination(
            captured_images, segmenter, classifier, preprocessor,
            all_seg_preprocess=all_seg_preprocess,
            all_cc_preprocess=all_cc_preprocess,
            seg_method=segmenter_preprocess,
            cc_method=cc_preprocess,
            force_centers=force_centers,
            segmenter_name=segmenter_name,
        )

        if best_cube_data is None:
            print("\nError: No valid preprocessing combination found. Aborting cube solve.")
            return

        cube_data = best_cube_data
        print(f"\nUsing: segmenter-preprocess={best_seg}, cc-preprocess={best_cc}")

        # Convert captured_images dict to list for display
        captured_images_list = [captured_images[face_key] for face_key in FACE_NAMES]

    else:
        # Original single-pass processing
        captured_images_list = []

        # Process each face
        for i, (face_key, face_display) in enumerate(zip(FACE_NAMES, FACE_DISPLAY_NAMES)):
            print(f"\n{'#' * 50}")
            print(f"  FACE {i+1}/6: {face_display}")
            print("#" * 50)

            image_path = face_files[face_key]

            # Load image for later display
            image = cv2.imread(image_path)
            if image is not None:
                captured_images_list.append(image.copy())
                captured_images[face_key] = image

            # Process the face
            classifications, facelets = process_single_face(image_path, segmenter, classifier, side_name=face_key, display=display,
                                                  segmenter_preprocess=segmenter_preprocess,
                                                  cc_preprocess=cc_preprocess, preprocessor=preprocessor)
            if classifications is None:
                print("\nError processing face. Aborting cube solve.")
                return

            # Store facelets for later display
            captured_facelets.append(facelets)

            # Get results
            face_string, face_array = print_classification_results(classifications, face_display)

            # Store for solver
            cube_data[face_key] = face_array

            print(f"\n{face_display} face captured successfully!")

            # Pause to review the displayed image if display mode is on
            if display:
                remaining = 6 - (i + 1)
                if remaining > 0:
                    print(f"\n{remaining} face(s) remaining. Press Enter to continue (or 'q' to cancel)...")
                else:
                    print("\nPress Enter to continue to solver (or 'q' to cancel)...")
                user_input = input("> ").strip().lower()
                dm = get_display_manager()
                dm.destroyAllWindows()
                dm.waitKey(1)  # Process any pending window events after destroy
                if user_input == 'q':
                    print("\nCancelled. Aborting cube solve.")
                    return
            else:
                # Show progress without pause
                remaining = 6 - (i + 1)
                if remaining > 0:
                    print(f"\n{remaining} face(s) remaining...")

        # Close any remaining display windows
        if display:
            dm = get_display_manager()
            dm.destroyAllWindows()
            dm.waitKey(1)  # Process any pending window events after destroy

    # Display all images with facelets if requested
    if display and captured_images_list:
        print("\nDisplaying all captured faces...")
        dm = get_display_manager()
        dm.display_images_grid(captured_images_list, labels=FACE_DISPLAY_NAMES,
                               window_name="All Cube Faces", cols=3,
                               facelets_list=captured_facelets if captured_facelets else None)

    # All faces captured - prepare solver input
    print("\n" + "=" * 50)
    print("  ALL FACES CAPTURED")
    print("=" * 50)

    # Display cube visualization
    counts_valid = print_cube_net(cube_data)

    # Check if color counts are valid
    if not counts_valid:
        print("ERROR: Invalid cube configuration - color counts are incorrect.")
        print("Each color must appear exactly 9 times on a valid Rubik's Cube.")
        input("Press Enter to return to menu...")
        return

    # Wait for user confirmation before solving
    user_input = input("Press Enter to solve, or 'q' to cancel: ").strip().lower()
    if user_input == 'q':
        print("Cancelled.")
        return

    # Apply orientation correction
    print("\nApplying orientation correction...")
    corrector = CubeOrientationCorrector(verbose=False)
    cube_data, corrections = corrector.correct(cube_data)
    if corrections:
        non_internal = {k: v for k, v in corrections.items() if not k.startswith('_')}
        if non_internal:
            print("Corrections applied:")
            for face, correction in non_internal.items():
                print(f"  {face}: {correction}")
        else:
            print("No corrections needed.")
    else:
        print("No corrections needed.")

    # Write JSON file for solver
    solver_input = {"cube": cube_data}
    json_path = "AStar_in.json"
    debug_print(f"Writing solver input to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(solver_input, f, indent=2)
    debug_print("Done writing solver input.")

    # Call the solver (full_cube_mode)
    print("\n" + "=" * 50)
    print("  SOLVING CUBE")
    print("=" * 50)

    debug_print("Initializing Kociemba solver...")
    start_time = time.time()
    with suppress_output():
        solver = KociembaSolver()
    solver_init_time = time.time() - start_time
    debug_print(f"Solver ready (took {solver_init_time:.3f}s)")

    print("\nRunning solver...")
    start_time = time.time()
    try:
        solver.RubikAStar()
        solve_time = time.time() - start_time

        # Read and display solution
        solution_path = "AStar_out.txt"
        if os.path.exists(solution_path):
            with open(solution_path, 'r') as f:
                solution = f.read().strip()

            print("\n" + "=" * 50)
            print("  SOLUTION FOUND!")
            print("=" * 50)
            print(f"\nMoves: {solution}")

            # Count moves
            moves = solution.split()
            print(f"Total moves: {len(moves)}")
            print(f"Solve time: {solve_time:.3f}s")

            # Animate solution if requested
            if animate and moves:
                animate_solution(cube_data, moves)
        else:
            print("\nError: Solution file not found.")

    except Exception as e:
        print(f"\nError running solver: {e}")


def camera_single_face_mode(display=False, segmenter_name: str = 'auto',
                            rotate: bool = False, segmenter_preprocess: str = None,
                            cc_preprocess: str = None, use_gpu: bool = True):
    """
    Mode 3: Capture a single face from camera and classify.

    Args:
        display: If True, show captured images on display
        segmenter_name: Name of segmentation algorithm to use
        rotate: If True, rotate captured images 180 degrees
        segmenter_preprocess: Preprocessing method name for segmentation
        cc_preprocess: Preprocessing method name for color classification
        use_gpu: If True, use GPU acceleration for preprocessing
    """
    if not JETSON_AVAILABLE:
        print("\nError: Camera mode requires Jetson hardware with IMX219 camera.")
        return

    print("\n" + "=" * 50)
    print("  CAMERA SINGLE FACE MODE")
    print("=" * 50)

    # Initialize segmenter
    debug_print(f"Initializing segmenter '{segmenter_name}'...")
    start_time = time.time()
    with suppress_output():
        segmenter = Segmenter.create(segmenter_name)
    segmenter_time = time.time() - start_time
    debug_print(f"Segmenter ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor if needed
    preprocessor = None
    if segmenter_preprocess or cc_preprocess:
        debug_print(f"Initializing GPUImagePreprocessor (GPU={'enabled' if use_gpu else 'disabled'})...")
        preprocessor = GPUImagePreprocessor(use_gpu=use_gpu)
        if segmenter_preprocess:
            debug_print(f"  Segmenter preprocessing: {segmenter_preprocess}")
        if cc_preprocess:
            debug_print(f"  Color classifier preprocessing: {cc_preprocess}")

    debug_print("Initializing JetsonCamera...")
    start_time = time.time()
    camera = JetsonCamera()
    if not camera.open():
        print("Error: Failed to open camera.")
        return
    camera_time = time.time() - start_time
    debug_print(f"JetsonCamera ready (took {camera_time:.3f}s)")

    try:
        # Instructions
        print("\n" + "-" * 50)
        print("Hold the Rubik's Cube face in front of the camera.")
        print("Make sure the face fills most of the frame.")
        print("Press Enter when ready, or 'q' to cancel...")
        print("-" * 50)

        user_input = input("> ").strip().lower()
        if user_input == 'q':
            print("Cancelled.")
            return

        # Capture with live preview (rotation applied in preview if enabled)
        image = camera.capture_with_preview(display=display, rotate=rotate)

        if image is None:
            print("Error: Failed to capture image.")
            return

        # Save the captured image
        capture_path = os.path.join("output_facelets", "camera_capture.jpg")
        os.makedirs("output_facelets", exist_ok=True)
        cv2.imwrite(capture_path, image)
        debug_print(f"Saved captured image to {capture_path}")

        # Process the image
        debug_print("Processing captured image...")
        classifications, facelets = process_image(image, segmenter, classifier, side_name="single", display=display,
                                        segmenter_preprocess=segmenter_preprocess,
                                        cc_preprocess=cc_preprocess, preprocessor=preprocessor)

        if classifications is None:
            return

        # Output results
        face_string, _ = print_classification_results(classifications)
        print(f"\nScan complete! Face string: {face_string}")

        # Close display window if it was opened
        if display:
            input("\nPress Enter to close the image display...")
            dm = get_display_manager()
            dm.destroyAllWindows()
            dm.waitKey(1)  # Process any pending window events after destroy

    finally:
        with suppress_output():
            camera.close()


def camera_full_cube_mode(display=False, segmenter_name: str = 'auto',
                          rotate: bool = False, segmenter_preprocess: str = None,
                          cc_preprocess: str = None, animate: bool = False,
                          all_seg_preprocess: bool = False, all_cc_preprocess: bool = False,
                          force_centers: bool = False, use_gpu: bool = True,
                          adaptive: bool = False):
    """
    Mode 4: Capture all 6 faces from camera and solve the cube.

    Args:
        display: If True, show captured images on display
        segmenter_name: Name of segmentation algorithm to use
        rotate: If True, rotate captured images 180 degrees
        segmenter_preprocess: Preprocessing method name for segmentation
        animate: If True, animate the solution moves after solving
        cc_preprocess: Preprocessing method name for color classification
        all_seg_preprocess: If True, try all segmentation preprocessing methods
        all_cc_preprocess: If True, try all CC preprocessing methods
        force_centers: If True, require centers to match expected colors
        use_gpu: If True, use GPU acceleration for preprocessing
        adaptive: If True, use adaptive evaluation with two-result confirmation
    """
    if not JETSON_AVAILABLE:
        print("\nError: Camera mode requires Jetson hardware with IMX219 camera.")
        return

    print("\n" + "=" * 50)
    print("  CAMERA FULL CUBE SOLVER MODE")
    print("=" * 50)
    print("\nYou will capture all 6 faces of the cube using the camera.")
    print("Follow the on-screen instructions for each face.")

    # Initialize segmenter
    debug_print(f"Initializing segmenter '{segmenter_name}'...")
    start_time = time.time()
    with suppress_output():
        segmenter = Segmenter.create(segmenter_name)
    segmenter_time = time.time() - start_time
    debug_print(f"Segmenter ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor (always needed for all_* modes)
    debug_print(f"Initializing GPUImagePreprocessor (GPU={'enabled' if use_gpu else 'disabled'})...")
    preprocessor = GPUImagePreprocessor(use_gpu=use_gpu)
    if segmenter_preprocess:
        debug_print(f"  Segmenter preprocessing: {segmenter_preprocess}")
    if cc_preprocess:
        debug_print(f"  Color classifier preprocessing: {cc_preprocess}")

    debug_print("Initializing JetsonCamera...")
    start_time = time.time()
    camera = JetsonCamera()
    if not camera.open():
        print("Error: Failed to open camera.")
        return
    camera_time = time.time() - start_time
    debug_print(f"JetsonCamera ready (took {camera_time:.3f}s)")

    # Dictionary to store face data and images
    cube_data = {}
    captured_images = {}  # Dict for multi-preprocess mode
    captured_images_list = []  # List for display
    captured_facelets = []

    # Check if we need to do multi-preprocessing evaluation
    use_multi_preprocess = all_seg_preprocess or all_cc_preprocess
    use_adaptive = adaptive and not use_multi_preprocess
    defer_processing = use_multi_preprocess or use_adaptive

    try:
        # Capture each face
        for i, (face_key, face_display) in enumerate(zip(FACE_NAMES, FACE_DISPLAY_NAMES)):
            print(f"\n{'#' * 50}")
            print(f"  FACE {i+1}/6: {face_display}")
            print("#" * 50)

            # Capture with live preview (rotation applied in preview if enabled)
            image = camera.capture_with_preview(display=display, rotate=rotate)

            if image is None:
                print("Error: Failed to capture image. Aborting.")
                return

            # Store for later processing and display
            captured_images[face_key] = image.copy()
            captured_images_list.append(image.copy())

            if not defer_processing:
                # Process immediately in single-pass mode
                debug_print("Processing captured image...")
                classifications, facelets = process_image(image, segmenter, classifier, side_name=face_key, display=display,
                                                segmenter_preprocess=segmenter_preprocess,
                                                cc_preprocess=cc_preprocess, preprocessor=preprocessor)

                if classifications is None:
                    print("\nError processing face. Aborting cube solve.")
                    return

                # Store facelets for later display
                captured_facelets.append(facelets)

                # Get results
                face_string, face_array = print_classification_results(classifications, face_display)

                # Store for solver
                cube_data[face_key] = face_array

            print(f"\n{face_display} face captured successfully!")

            # Pause to review the displayed image if display mode is on
            if display:
                remaining = 6 - (i + 1)
                if remaining > 0:
                    print(f"\n{remaining} face(s) remaining. Press Enter to continue (or 'q' to cancel)...")
                else:
                    print("\nPress Enter to continue to solver (or 'q' to cancel)...")
                user_input = input("> ").strip().lower()
                dm = get_display_manager()
                dm.destroyAllWindows()
                dm.waitKey(1)  # Process any pending window events
                if user_input == 'q':
                    print("\nCancelled. Aborting cube solve.")
                    return
            else:
                # Show progress without pause
                remaining = 6 - (i + 1)
                if remaining > 0:
                    print(f"\n{remaining} face(s) remaining...")

    finally:
        with suppress_output():
            camera.close()

    # If adaptive mode, use adaptive evaluation
    if use_adaptive:
        print("\nRunning adaptive evaluation (two-result confirmation)...")
        evaluator = AdaptiveEvaluator()
        result = evaluator.evaluate(
            captured_images, classifier, preprocessor,
            max_attempts=100,
            combos_per_round=5,
            background_samples=10,
            force_centers=force_centers,
            record_metrics=True,
            verbose=True
        )

        if not result['is_valid']:
            if result.get('valid_results_count', 0) > 0:
                print(f"\nWARNING: Found {result['valid_results_count']} valid result(s) but could not confirm.")
                print("No two results matched - cube state may be ambiguous.")
            else:
                print("\nError: No valid result found. Aborting cube solve.")
            return

        cube_data = result['cube_data']

    # If multi-preprocess mode, evaluate all combinations now
    elif use_multi_preprocess:
        best_cube_data, best_confidences, best_seg, best_cc, all_results = find_best_preprocessing_combination(
            captured_images, segmenter, classifier, preprocessor,
            all_seg_preprocess=all_seg_preprocess,
            all_cc_preprocess=all_cc_preprocess,
            seg_method=segmenter_preprocess,
            cc_method=cc_preprocess,
            force_centers=force_centers,
            segmenter_name=segmenter_name,
        )

        if best_cube_data is None:
            print("\nError: No valid preprocessing combination found. Aborting cube solve.")
            return

        cube_data = best_cube_data
        print(f"\nUsing: segmenter-preprocess={best_seg}, cc-preprocess={best_cc}")

    # All faces captured - display all images if requested
    if display and captured_images_list:
        print("\nDisplaying all captured faces...")
        dm = get_display_manager()
        dm.display_images_grid(captured_images_list, labels=FACE_DISPLAY_NAMES,
                               window_name="All Cube Faces", cols=3,
                               facelets_list=captured_facelets if captured_facelets else None)

    # Prepare solver input
    print("\n" + "=" * 50)
    print("  ALL FACES CAPTURED")
    print("=" * 50)

    # Display cube visualization
    counts_valid = print_cube_net(cube_data)

    # Check if color counts are valid
    if not counts_valid:
        print("ERROR: Invalid cube configuration - color counts are incorrect.")
        print("Each color must appear exactly 9 times on a valid Rubik's Cube.")
        input("Press Enter to return to menu...")
        return

    # Wait for user confirmation before solving
    user_input = input("Press Enter to solve, or 'q' to cancel: ").strip().lower()
    if user_input == 'q':
        print("Cancelled.")
        return

    # Apply orientation correction
    print("\nApplying orientation correction...")
    corrector = CubeOrientationCorrector(verbose=False)
    cube_data, corrections = corrector.correct(cube_data)
    if corrections:
        non_internal = {k: v for k, v in corrections.items() if not k.startswith('_')}
        if non_internal:
            print("Corrections applied:")
            for face, correction in non_internal.items():
                print(f"  {face}: {correction}")
        else:
            print("No corrections needed.")
    else:
        print("No corrections needed.")

    # Write JSON file for solver
    solver_input = {"cube": cube_data}
    json_path = "AStar_in.json"
    debug_print(f"Writing solver input to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(solver_input, f, indent=2)
    debug_print("Done writing solver input.")

    # Call the solver (camera_full_cube_mode)
    print("\n" + "=" * 50)
    print("  SOLVING CUBE")
    print("=" * 50)

    debug_print("Initializing Kociemba solver...")
    start_time = time.time()
    with suppress_output():
        solver = KociembaSolver()
    solver_init_time = time.time() - start_time
    debug_print(f"Solver ready (took {solver_init_time:.3f}s)")

    print("\nRunning solver...")
    start_time = time.time()
    try:
        solver.RubikAStar()
        solve_time = time.time() - start_time

        # Read and display solution
        solution_path = "AStar_out.txt"
        if os.path.exists(solution_path):
            with open(solution_path, 'r') as f:
                solution = f.read().strip()

            print("\n" + "=" * 50)
            print("  SOLUTION FOUND!")
            print("=" * 50)
            print(f"\nMoves: {solution}")

            # Count moves
            moves = solution.split()
            print(f"Total moves: {len(moves)}")
            print(f"Solve time: {solve_time:.3f}s")

            # Animate solution if requested
            if animate and moves:
                animate_solution(cube_data, moves)
        else:
            print("\nError: Solution file not found.")

    except Exception as e:
        print(f"\nError running solver: {e}")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Rubik's Cube Scanner & Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Show captured images on display (for Jetson with monitor)'
    )
    parser.add_argument(
        '--segmenter',
        type=str,
        default=Segmenter.get_default(),
        metavar='NAME',
        help=f'Segmentation algorithm to use (default: {Segmenter.get_default()}). '
             f'Options: {", ".join(Segmenter.get_available_segmenters())}'
    )
    parser.add_argument(
        '--rotate',
        action='store_true',
        help='Rotate camera images 180 degrees (for inverted camera mounting)'
    )
    parser.add_argument(
        '--segmenter-preprocess',
        type=str,
        default=None,
        metavar='METHOD',
        help='Preprocess image before segmentation (e.g., satboost, bilateral, clahe-lab)'
    )
    parser.add_argument(
        '--cc-preprocess',
        type=str,
        default=None,
        metavar='METHOD',
        help='Preprocess image before color classification (e.g., satboost, bilateral, clahe-lab)'
    )
    parser.add_argument(
        '--no-animation',
        action='store_true',
        help='Disable solution animation (animation is on by default when --display is enabled)'
    )
    parser.add_argument(
        '--all-segmenter-preprocess',
        action='store_true',
        help='Try all preprocessing methods for segmentation and select best result'
    )
    parser.add_argument(
        '--all-cc-preprocess',
        action='store_true',
        help='Try all preprocessing methods for color classification and select best result'
    )
    parser.add_argument(
        '--force-centers',
        action='store_true',
        help='Force center facelets to match expected colors (Y/W/B/G/O/R for up/down/front/back/left/right)'
    )
    parser.add_argument(
        '--nogpu',
        action='store_true',
        help='Disable GPU acceleration for preprocessing (use CPU only)'
    )
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Use adaptive evaluation: intelligently selects preprocessing combos based on '
             'historical metrics, requires two identical valid results for confirmation'
    )
    args = parser.parse_args()

    # Validate preprocessing options - auto-detect GPU availability
    use_gpu = False
    gpu_status_msg = None
    if args.nogpu:
        preprocessor = ImagePreprocessor()
        gpu_status_msg = "GPU: Disabled (--nogpu flag)"
    elif not GPU_PREPROCESSOR_AVAILABLE:
        preprocessor = ImagePreprocessor()
        gpu_status_msg = "GPU: Not available (GPUImagePreprocessor module not found)"
    else:
        # Try to initialize GPU preprocessor and check if GPU is actually available
        preprocessor = GPUImagePreprocessor(use_gpu=True)
        if preprocessor.is_gpu_enabled():
            use_gpu = True
            gpu_status_msg = "GPU: Enabled (VPI/CUDA)"
        else:
            # GPU not available, fall back to CPU
            preprocessor = ImagePreprocessor()
            gpu_status_msg = "GPU: Not available (VPI/CUDA not detected)"
    valid_methods = preprocessor.get_available_methods()
    valid_methods_normalized = [m.lower().replace('_', '-') for m in valid_methods]

    def normalize_method(method):
        """Normalize method name: lowercase and replace underscores with dashes."""
        return method.lower().replace('_', '-') if method else None

    if args.segmenter_preprocess and normalize_method(args.segmenter_preprocess) not in valid_methods_normalized:
        print(f"Error: Invalid segmenter preprocessing method: '{args.segmenter_preprocess}'")
        print(f"\nAvailable methods:")
        for method in valid_methods:
            desc = preprocessor.get_method_description(method)
            print(f"  {method:<20} - {desc}")
        sys.exit(1)

    if args.cc_preprocess and normalize_method(args.cc_preprocess) not in valid_methods_normalized:
        print(f"Error: Invalid color classifier preprocessing method: '{args.cc_preprocess}'")
        print(f"\nAvailable methods:")
        for method in valid_methods:
            desc = preprocessor.get_method_description(method)
            print(f"  {method:<20} - {desc}")
        sys.exit(1)

    print("=" * 50)
    print("  RUBIK'S CUBE SCANNER & SOLVER")
    print("  Segmentation + Color Classification + IDA* Solver")
    print("=" * 50)

    # Show platform and GPU status
    if JETSON_AVAILABLE:
        print("\n[Jetson detected - Camera modes available]")
        if args.display:
            print("[Display mode enabled - Images will be shown on monitor]")
    else:
        print("\n[Running on non-Jetson platform - File modes only]")

    # Show GPU status
    print(f"[{gpu_status_msg}]")

    # Validate and show segmenter
    if not Segmenter.is_valid(args.segmenter):
        print(f"Error: Unknown segmenter '{args.segmenter}'")
        print(f"Available segmenters: {', '.join(Segmenter.get_available_segmenters())}")
        return
    segmenter_desc = Segmenter.get_description(args.segmenter)
    print(f"[Segmenter: {args.segmenter} - {segmenter_desc}]")

    if args.rotate:
        print("[Camera rotation enabled - images will be rotated 180 degrees]")

    # Show preprocessing options if set
    if args.segmenter_preprocess:
        print(f"[Segmenter preprocessing: {args.segmenter_preprocess}]")
    if args.cc_preprocess:
        print(f"[Color classifier preprocessing: {args.cc_preprocess}]")
    if args.adaptive:
        print("[Adaptive mode: Uses historical metrics with two-result confirmation]")

    # Menu-driven loop
    while True:
        print("\n" + "-" * 50)
        print("Select Mode:")
        print("  1. Single Face (File)  - Load image and classify colors")
        print("  2. Full Cube (File)    - Load 6 images and solve cube")
        if JETSON_AVAILABLE:
            print("  3. Single Face (Camera) - Capture and classify one face")
            print("  4. Full Cube (Camera)   - Capture 6 faces and solve")
        print("  q. Quit")
        print("-" * 50)

        choice = input("> ").strip().lower()

        if choice == '1':
            single_face_mode(segmenter_name=args.segmenter,
                            segmenter_preprocess=args.segmenter_preprocess,
                            cc_preprocess=args.cc_preprocess, use_gpu=use_gpu)
        elif choice == '2':
            # Animation is on by default when display is enabled
            animate = args.display and not args.no_animation
            full_cube_mode(segmenter_name=args.segmenter, display=args.display,
                          segmenter_preprocess=args.segmenter_preprocess,
                          cc_preprocess=args.cc_preprocess, animate=animate,
                          all_seg_preprocess=args.all_segmenter_preprocess,
                          all_cc_preprocess=args.all_cc_preprocess,
                          force_centers=args.force_centers, use_gpu=use_gpu,
                          adaptive=args.adaptive)
        elif choice == '3' and JETSON_AVAILABLE:
            camera_single_face_mode(display=args.display, segmenter_name=args.segmenter,
                                   rotate=args.rotate,
                                   segmenter_preprocess=args.segmenter_preprocess,
                                   cc_preprocess=args.cc_preprocess, use_gpu=use_gpu)
        elif choice == '4' and JETSON_AVAILABLE:
            # Animation is on by default when display is enabled
            animate = args.display and not args.no_animation
            camera_full_cube_mode(display=args.display, segmenter_name=args.segmenter,
                                 rotate=args.rotate,
                                 segmenter_preprocess=args.segmenter_preprocess,
                                 cc_preprocess=args.cc_preprocess, animate=animate,
                                 all_seg_preprocess=args.all_segmenter_preprocess,
                                 all_cc_preprocess=args.all_cc_preprocess,
                                 force_centers=args.force_centers, use_gpu=use_gpu,
                                 adaptive=args.adaptive)
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            valid_choices = "1, 2, 3, 4, or q" if JETSON_AVAILABLE else "1, 2, or q"
            print(f"Invalid choice. Please enter {valid_choices}.")


if __name__ == "__main__":
    main()
