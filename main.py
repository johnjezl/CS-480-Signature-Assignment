"""
Rubik's Cube Face Scanner - Main Application

Menu-driven application with multiple modes:
1. Single Face Mode: Scan one image file and classify colors
2. Full Cube Mode: Scan all 6 faces from files and solve the cube
3. Camera Single Face: Capture one face from camera and classify (Jetson only)
4. Camera Full Cube: Capture all 6 faces from camera and solve (Jetson only)

Usage:
    python main.py [--display] [--v2] [--v3] [--v4] [--v5] [--rotate]
                   [--segmenter-preprocess METHOD] [--cc-preprocess METHOD]

Options:
    --display    Show captured images on display (for Jetson with monitor)
    --v2         Use v2 segmenter with improved detection algorithms
    --v3         Use v3 segmenter with contour-based facelet detection
    --v4         Use v4 segmenter with OpenCV square detection (Canny + contours)
    --v5         Use v5 segmenter with brightness-based Otsu thresholding (Greg's CV)
    --rotate     Rotate camera images 180 degrees (for inverted camera mounting)
    --segmenter-preprocess METHOD   Preprocess image before segmentation
    --cc-preprocess METHOD          Preprocess image before color classification

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

from facelet_segmenter import FaceletSegmenter
from facelet_segmenter_v2 import FaceletSegmenterV2
from facelet_segmenter_v3 import FaceletSegmenterV3
from facelet_segmenter_v4 import FaceletSegmenterV4
from facelet_segmenter_v5 import FaceletSegmenterV5
from facelet_segmenter_auto import FaceletSegmenterAuto
from FaceletColorClassifier import FaceletColorClassifier
from IDASolver import IDASolver, KociembaSolver
from ImagePreprocessor import ImagePreprocessor

# Try to import Jetson camera module
try:
    from JetsonCamera import JetsonCamera, is_jetson, display_image, display_images_grid, debug_print, suppress_output
    JETSON_AVAILABLE = is_jetson()
except ImportError:
    JETSON_AVAILABLE = False
    JetsonCamera = None

    def is_jetson():
        return False

    def display_image(image, window_name="Image", wait_key=True):
        pass

    def display_images_grid(images, labels=None, window_name="Cube Faces", cols=3):
        pass

    def debug_print(msg):
        pass

    from contextlib import contextmanager
    @contextmanager
    def suppress_output():
        yield


# Color abbreviation map
COLOR_ABBREV = {
    'white': 'W', 'yellow': 'Y', 'red': 'R',
    'orange': 'O', 'blue': 'B', 'green': 'G'
}

# Face names for the solver (in order of input)
FACE_NAMES = ['up', 'down', 'front', 'back', 'left', 'right']
FACE_DISPLAY_NAMES = ['Up (Yellow)', 'Down (White)', 'Front (Blue)',
                      'Back (Green)', 'Left (Orange)', 'Right (Red)']


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


def display_face_and_facelets(image, facelets, window_name="Face and Facelets"):
    """
    Display the face image alongside a 3x3 grid of extracted facelets.

    Args:
        image: BGR numpy array of the full face image
        facelets: numpy array of shape (3, 3, 64, 64, 3)
        window_name: Name for the display window
    """
    cell_size = 100  # Size of each facelet display
    border = 2       # White border around each facelet
    spacing = 2      # Space between facelets

    bordered_size = cell_size + border * 2
    grid_size = bordered_size * 3 + spacing * 2

    # Black background for facelet grid
    facelet_grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    for row in range(3):
        for col in range(3):
            facelet = facelets[row, col]
            # Resize facelet to cell size
            facelet_resized = cv2.resize(facelet, (cell_size, cell_size))
            # Add white border around facelet
            bordered = cv2.copyMakeBorder(facelet_resized, border, border, border, border,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
            # Place in grid
            y1 = row * (bordered_size + spacing)
            x1 = col * (bordered_size + spacing)
            facelet_grid[y1:y1+bordered_size, x1:x1+bordered_size] = bordered

    # Scale face to match facelet grid height
    scale = grid_size / image.shape[0]
    new_width = int(image.shape[1] * scale)
    face_scaled = cv2.resize(image, (new_width, grid_size))

    # Combine face and facelet grid side by side
    gap = spacing
    gap_img = np.zeros((grid_size, gap, 3), dtype=np.uint8)
    combined = np.hstack([face_scaled, gap_img, facelet_grid])

    # Display - destroy any existing window first to avoid showing stale content
    with suppress_output():
        try:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)  # Process the destroy
        except cv2.error:
            pass  # Window didn't exist yet, that's fine
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, combined)
        cv2.waitKey(100)  # Give more time for window to update


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
        with suppress_output():
            display_face_and_facelets(seg_image, facelets, window_name)

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


def single_face_mode(use_v2: bool = False, use_v3: bool = False, use_v4: bool = False,
                     use_v5: bool = False, use_auto: bool = False,
                     segmenter_preprocess: str = None, cc_preprocess: str = None):
    """Mode 1: Process a single face image."""
    print("\n" + "=" * 50)
    print("  SINGLE FACE MODE")
    print("=" * 50)

    # Initialize components with timing
    if use_auto:
        segmenter_name = "FaceletSegmenterAuto"
        segmenter_class = FaceletSegmenterAuto
    elif use_v5:
        segmenter_name = "FaceletSegmenterV5"
        segmenter_class = FaceletSegmenterV5
    elif use_v4:
        segmenter_name = "FaceletSegmenterV4"
        segmenter_class = FaceletSegmenterV4
    elif use_v3:
        segmenter_name = "FaceletSegmenterV3"
        segmenter_class = FaceletSegmenterV3
    elif use_v2:
        segmenter_name = "FaceletSegmenterV2"
        segmenter_class = FaceletSegmenterV2
    else:
        segmenter_name = "FaceletSegmenter"
        segmenter_class = FaceletSegmenter

    debug_print(f"Initializing {segmenter_name}...")
    start_time = time.time()
    with suppress_output():
        segmenter = segmenter_class()
    segmenter_time = time.time() - start_time
    debug_print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor if needed
    preprocessor = None
    if segmenter_preprocess or cc_preprocess:
        debug_print("Initializing ImagePreprocessor...")
        preprocessor = ImagePreprocessor()
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


def full_cube_mode(use_v2: bool = False, use_v3: bool = False, use_v4: bool = False,
                   use_v5: bool = False, use_auto: bool = False, display: bool = False,
                   segmenter_preprocess: str = None, cc_preprocess: str = None):
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

    # Initialize components with timing
    if use_auto:
        segmenter_name = "FaceletSegmenterAuto"
        segmenter_class = FaceletSegmenterAuto
    elif use_v5:
        segmenter_name = "FaceletSegmenterV5"
        segmenter_class = FaceletSegmenterV5
    elif use_v4:
        segmenter_name = "FaceletSegmenterV4"
        segmenter_class = FaceletSegmenterV4
    elif use_v3:
        segmenter_name = "FaceletSegmenterV3"
        segmenter_class = FaceletSegmenterV3
    elif use_v2:
        segmenter_name = "FaceletSegmenterV2"
        segmenter_class = FaceletSegmenterV2
    else:
        segmenter_name = "FaceletSegmenter"
        segmenter_class = FaceletSegmenter

    debug_print(f"Initializing {segmenter_name}...")
    start_time = time.time()
    with suppress_output():
        segmenter = segmenter_class()
    segmenter_time = time.time() - start_time
    debug_print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor if needed
    preprocessor = None
    if segmenter_preprocess or cc_preprocess:
        debug_print("Initializing ImagePreprocessor...")
        preprocessor = ImagePreprocessor()
        if segmenter_preprocess:
            debug_print(f"  Segmenter preprocessing: {segmenter_preprocess}")
        if cc_preprocess:
            debug_print(f"  Color classifier preprocessing: {cc_preprocess}")

    # Dictionary to store face data for the solver
    cube_data = {}
    captured_images = []
    captured_facelets = []

    # Process each face
    for i, (face_key, face_display) in enumerate(zip(FACE_NAMES, FACE_DISPLAY_NAMES)):
        print(f"\n{'#' * 50}")
        print(f"  FACE {i+1}/6: {face_display}")
        print("#" * 50)

        image_path = face_files[face_key]

        # Load image for later display
        image = cv2.imread(image_path)
        if image is not None:
            captured_images.append(image.copy())

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
            with suppress_output():
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Process any pending window events after destroy
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
        with suppress_output():
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process any pending window events after destroy

    # Display all images with facelets if requested
    if display and captured_images:
        print("\nDisplaying all captured faces...")
        display_images_grid(captured_images, labels=FACE_DISPLAY_NAMES,
                            window_name="All Cube Faces", cols=3, facelets_list=captured_facelets)

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

    # Write JSON file for solver
    solver_input = {"cube": cube_data}
    json_path = "AStar_in.json"
    debug_print(f"Writing solver input to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(solver_input, f, indent=2)
    debug_print("Done writing solver input.")

    # Call the solver
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
        else:
            print("\nError: Solution file not found.")

    except Exception as e:
        print(f"\nError running solver: {e}")


def camera_single_face_mode(display=False, use_v2: bool = False, use_v3: bool = False,
                            use_v4: bool = False, use_v5: bool = False, use_auto: bool = False,
                            rotate: bool = False, segmenter_preprocess: str = None,
                            cc_preprocess: str = None):
    """
    Mode 3: Capture a single face from camera and classify.

    Args:
        display: If True, show captured images on display
        use_v2: If True, use v2 segmenter with improved detection
        use_v3: If True, use v3 segmenter with contour-based detection
        use_v4: If True, use v4 segmenter with OpenCV square detection
        use_v5: If True, use v5 segmenter with brightness-based Otsu thresholding
        rotate: If True, rotate captured images 180 degrees
        segmenter_preprocess: Preprocessing method name for segmentation
        cc_preprocess: Preprocessing method name for color classification
    """
    if not JETSON_AVAILABLE:
        print("\nError: Camera mode requires Jetson hardware with IMX219 camera.")
        return

    print("\n" + "=" * 50)
    print("  CAMERA SINGLE FACE MODE")
    print("=" * 50)

    # Initialize components with timing
    if use_auto:
        segmenter_name = "FaceletSegmenterAuto"
        segmenter_class = FaceletSegmenterAuto
    elif use_v5:
        segmenter_name = "FaceletSegmenterV5"
        segmenter_class = FaceletSegmenterV5
    elif use_v4:
        segmenter_name = "FaceletSegmenterV4"
        segmenter_class = FaceletSegmenterV4
    elif use_v3:
        segmenter_name = "FaceletSegmenterV3"
        segmenter_class = FaceletSegmenterV3
    elif use_v2:
        segmenter_name = "FaceletSegmenterV2"
        segmenter_class = FaceletSegmenterV2
    else:
        segmenter_name = "FaceletSegmenter"
        segmenter_class = FaceletSegmenter

    debug_print(f"Initializing {segmenter_name}...")
    start_time = time.time()
    with suppress_output():
        segmenter = segmenter_class()
    segmenter_time = time.time() - start_time
    debug_print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor if needed
    preprocessor = None
    if segmenter_preprocess or cc_preprocess:
        debug_print("Initializing ImagePreprocessor...")
        preprocessor = ImagePreprocessor()
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
            with suppress_output():
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Process any pending window events after destroy

    finally:
        with suppress_output():
            camera.close()


def camera_full_cube_mode(display=False, use_v2: bool = False, use_v3: bool = False,
                          use_v4: bool = False, use_v5: bool = False, use_auto: bool = False,
                          rotate: bool = False, segmenter_preprocess: str = None,
                          cc_preprocess: str = None):
    """
    Mode 4: Capture all 6 faces from camera and solve the cube.

    Args:
        display: If True, show captured images on display
        use_v2: If True, use v2 segmenter with improved detection
        use_v3: If True, use v3 segmenter with contour-based detection
        use_v4: If True, use v4 segmenter with OpenCV square detection
        use_v5: If True, use v5 segmenter with brightness-based Otsu thresholding
        rotate: If True, rotate captured images 180 degrees
        segmenter_preprocess: Preprocessing method name for segmentation
        cc_preprocess: Preprocessing method name for color classification
    """
    if not JETSON_AVAILABLE:
        print("\nError: Camera mode requires Jetson hardware with IMX219 camera.")
        return

    print("\n" + "=" * 50)
    print("  CAMERA FULL CUBE SOLVER MODE")
    print("=" * 50)
    print("\nYou will capture all 6 faces of the cube using the camera.")
    print("Follow the on-screen instructions for each face.")

    # Initialize components with timing
    if use_auto:
        segmenter_name = "FaceletSegmenterAuto"
        segmenter_class = FaceletSegmenterAuto
    elif use_v5:
        segmenter_name = "FaceletSegmenterV5"
        segmenter_class = FaceletSegmenterV5
    elif use_v4:
        segmenter_name = "FaceletSegmenterV4"
        segmenter_class = FaceletSegmenterV4
    elif use_v3:
        segmenter_name = "FaceletSegmenterV3"
        segmenter_class = FaceletSegmenterV3
    elif use_v2:
        segmenter_name = "FaceletSegmenterV2"
        segmenter_class = FaceletSegmenterV2
    else:
        segmenter_name = "FaceletSegmenter"
        segmenter_class = FaceletSegmenter

    debug_print(f"Initializing {segmenter_name}...")
    start_time = time.time()
    with suppress_output():
        segmenter = segmenter_class()
    segmenter_time = time.time() - start_time
    debug_print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Initialize preprocessor if needed
    preprocessor = None
    if segmenter_preprocess or cc_preprocess:
        debug_print("Initializing ImagePreprocessor...")
        preprocessor = ImagePreprocessor()
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

    # Face orientation instructions
    FACE_INSTRUCTIONS = {
        'up': "Hold the cube with the YELLOW center facing the camera.\n"
              "         The GREEN face should be at the top.",
        'down': "Hold the cube with the WHITE center facing the camera.\n"
                "         The BLUE face should be at the top.",
        'front': "Hold the cube with the BLUE center facing the camera.\n"
                 "         The YELLOW face should be at the top.",
        'back': "Hold the cube with the GREEN center facing the camera.\n"
                "         The YELLOW face should be at the top.",
        'left': "Hold the cube with the ORANGE center facing the camera.\n"
                "         The YELLOW face should be at the top.",
        'right': "Hold the cube with the RED center facing the camera.\n"
                 "         The YELLOW face should be at the top."
    }

    # Dictionary to store face data and images
    cube_data = {}
    captured_images = []
    captured_facelets = []

    try:
        # Process each face
        for i, (face_key, face_display) in enumerate(zip(FACE_NAMES, FACE_DISPLAY_NAMES)):
            print(f"\n{'#' * 50}")
            print(f"  FACE {i+1}/6: {face_display}")
            print("#" * 50)

            # Show orientation instructions
            print(f"\nOrientation: {FACE_INSTRUCTIONS[face_key]}")

            # Capture with live preview (rotation applied in preview if enabled)
            image = camera.capture_with_preview(display=display, rotate=rotate)

            if image is None:
                print("Error: Failed to capture image. Aborting.")
                return

            # Store for later display
            captured_images.append(image.copy())

            # Process the image
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
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Process any pending window events
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

    # All faces captured - display all images if requested
    if display and captured_images:
        print("\nDisplaying all captured faces...")
        display_images_grid(captured_images, labels=FACE_DISPLAY_NAMES,
                            window_name="All Cube Faces", cols=3, facelets_list=captured_facelets)

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

    # Write JSON file for solver
    solver_input = {"cube": cube_data}
    json_path = "AStar_in.json"
    debug_print(f"Writing solver input to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(solver_input, f, indent=2)
    debug_print("Done writing solver input.")

    # Call the solver
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
        else:
            print("\nError: Solution file not found.")

    except Exception as e:
        print(f"\nError running solver: {e}")


def camera_segmenter_preprocess_comparison_mode(display=True, use_v2: bool = False, use_v3: bool = False,
                                                 use_v4: bool = False, use_v5: bool = False, use_auto: bool = False,
                                                 rotate: bool = False):
    """
    Mode 5: Capture a single face and compare all preprocessing methods for segmentation.

    Displays a tiled grid showing the segmentation result for each preprocessing
    method, with the method name overlaid on each tile.

    Args:
        display: If True, show the comparison grid on display
        use_v2: If True, use v2 segmenter
        use_v3: If True, use v3 segmenter
        use_v4: If True, use v4 segmenter
        use_v5: If True, use v5 segmenter
        use_auto: If True, use auto-selecting segmenter
        rotate: If True, rotate captured images 180 degrees
    """
    if not JETSON_AVAILABLE:
        print("\nError: Camera mode requires Jetson hardware with IMX219 camera.")
        return

    print("\n" + "=" * 50)
    print("  SEGMENTER PREPROCESSOR COMPARISON MODE")
    print("=" * 50)
    print("\nThis mode captures one face and shows segmentation results")
    print("for all preprocessing methods side-by-side.")

    # Initialize segmenter
    if use_auto:
        segmenter_name = "FaceletSegmenterAuto"
        segmenter_class = FaceletSegmenterAuto
    elif use_v5:
        segmenter_name = "FaceletSegmenterV5"
        segmenter_class = FaceletSegmenterV5
    elif use_v4:
        segmenter_name = "FaceletSegmenterV4"
        segmenter_class = FaceletSegmenterV4
    elif use_v3:
        segmenter_name = "FaceletSegmenterV3"
        segmenter_class = FaceletSegmenterV3
    elif use_v2:
        segmenter_name = "FaceletSegmenterV2"
        segmenter_class = FaceletSegmenterV2
    else:
        segmenter_name = "FaceletSegmenter"
        segmenter_class = FaceletSegmenter

    debug_print(f"Initializing {segmenter_name}...")
    start_time = time.time()
    with suppress_output():
        segmenter = segmenter_class()
    segmenter_time = time.time() - start_time
    debug_print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    debug_print("Initializing ImagePreprocessor...")
    preprocessor = ImagePreprocessor()
    preprocess_methods = preprocessor.get_available_methods()
    debug_print(f"Found {len(preprocess_methods)} preprocessing methods")

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

        # Capture with live preview
        image = camera.capture_with_preview(display=display, rotate=rotate)

        if image is None:
            print("Error: Failed to capture image.")
            return

        print(f"\nCaptured image: {image.shape[1]}x{image.shape[0]}")
        print(f"\nProcessing {len(preprocess_methods)} preprocessing methods...")

        # Process each preprocessing method
        results = []  # List of (method_name, facelets, classifications, avg_conf)

        for i, method in enumerate(preprocess_methods):
            print(f"  [{i+1}/{len(preprocess_methods)}] {method}...", end=" ", flush=True)

            # Apply preprocessing (start from original)
            if method.lower() != 'none':
                processed = preprocessor.apply(method, image)
            else:
                processed = image.copy()

            # Segment
            facelets = segmenter.segment(processed)

            # Classify
            classifications = classifier.classify_face(facelets)

            # Calculate average confidence
            total_conf = 0
            for row in range(3):
                for col in range(3):
                    _, conf = classifications[row, col]
                    total_conf += conf
            avg_conf = total_conf / 9

            results.append((method, facelets, classifications, avg_conf))
            print(f"{avg_conf:.1f}%")

        # Sort by confidence for ranking
        sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

        # Print ranking
        print("\n" + "=" * 50)
        print("PREPROCESSING COMPARISON RESULTS")
        print("=" * 50)
        print(f"\n{'Rank':<5} {'Method':<20} {'Avg Confidence':<15}")
        print("-" * 40)
        for rank, (method, _, _, avg_conf) in enumerate(sorted_results, 1):
            print(f"{rank:<5} {method:<20} {avg_conf:>10.1f}%")

        # Create comparison grid image
        # Layout: 5 columns (fits 22 methods in 5 rows)
        cols = 5
        rows = (len(results) + cols - 1) // cols

        # Each tile: preprocessed image (resized) + 3x3 facelet grid + label
        tile_w = 200
        tile_h = 180
        facelet_size = 40

        grid_w = cols * tile_w
        grid_h = rows * tile_h + 40  # Extra space for title

        comparison_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        comparison_grid[:] = (40, 40, 40)  # Dark gray background

        # Title
        cv2.putText(comparison_grid, f"Preprocessing Comparison ({segmenter_name})",
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Color map for display
        COLOR_BGR = {
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0)
        }

        for idx, (method, facelets, classifications, avg_conf) in enumerate(results):
            row = idx // cols
            col = idx % cols

            x_base = col * tile_w
            y_base = row * tile_h + 40  # Offset for title

            # Draw preprocessed image (small thumbnail)
            if method.lower() != 'none':
                processed = preprocessor.apply(method, image)
            else:
                processed = image.copy()

            thumb_h = 80
            thumb_w = int(processed.shape[1] * thumb_h / processed.shape[0])
            if thumb_w > tile_w - 10:
                thumb_w = tile_w - 10
                thumb_h = int(processed.shape[0] * thumb_w / processed.shape[1])

            thumb = cv2.resize(processed, (thumb_w, thumb_h))

            # Center the thumbnail
            thumb_x = x_base + (tile_w - thumb_w) // 2
            thumb_y = y_base + 5

            comparison_grid[thumb_y:thumb_y+thumb_h, thumb_x:thumb_x+thumb_w] = thumb

            # Draw 3x3 color grid below thumbnail
            grid_start_x = x_base + (tile_w - 3 * facelet_size) // 2
            grid_start_y = thumb_y + thumb_h + 5

            for r in range(3):
                for c in range(3):
                    color_name, conf = classifications[r, c]
                    bgr = COLOR_BGR.get(color_name, (128, 128, 128))

                    fx = grid_start_x + c * facelet_size
                    fy = grid_start_y + r * facelet_size

                    # Fill color square
                    cv2.rectangle(comparison_grid, (fx+1, fy+1),
                                 (fx+facelet_size-1, fy+facelet_size-1), bgr, -1)
                    # Border
                    cv2.rectangle(comparison_grid, (fx, fy),
                                 (fx+facelet_size, fy+facelet_size), (80, 80, 80), 1)

            # Method name and confidence label at bottom
            label = f"{method}"
            conf_label = f"{avg_conf:.1f}%"

            # Find rank for color coding
            rank = next(i for i, (m, _, _, _) in enumerate(sorted_results, 1) if m == method)
            if rank <= 3:
                label_color = (0, 255, 0)  # Green for top 3
            elif rank <= 6:
                label_color = (0, 255, 255)  # Yellow for top 6
            else:
                label_color = (200, 200, 200)  # Gray for rest

            label_y = grid_start_y + 3 * facelet_size + 15
            cv2.putText(comparison_grid, label, (x_base + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)
            cv2.putText(comparison_grid, conf_label, (x_base + tile_w - 45, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)

        # Save the comparison grid
        output_path = os.path.join("output_facelets", "preprocess_comparison.jpg")
        os.makedirs("output_facelets", exist_ok=True)
        cv2.imwrite(output_path, comparison_grid)
        print(f"\nSaved comparison grid to {output_path}")

        # Display if requested
        if display:
            # Get screen dimensions and scale grid to fill the screen
            try:
                # Try to get screen size from X display
                import subprocess
                result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'dimensions:' in line:
                        dims = line.split()[1].split('x')
                        screen_w, screen_h = int(dims[0]), int(dims[1])
                        break
                else:
                    # Fallback to common Jetson display resolution
                    screen_w, screen_h = 1920, 1080
            except Exception:
                # Fallback to common Jetson display resolution
                screen_w, screen_h = 1920, 1080

            # Calculate scale factor to fill screen while maintaining aspect ratio
            scale_w = screen_w / grid_w
            scale_h = screen_h / grid_h
            scale = min(scale_w, scale_h) * 0.95  # 95% of screen for some margin

            # Scale the comparison grid
            new_w = int(grid_w * scale)
            new_h = int(grid_h * scale)
            scaled_grid = cv2.resize(comparison_grid, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            print(f"\nDisplaying at {new_w}x{new_h} (scaled from {grid_w}x{grid_h})")
            print("Press Enter to close the display...")

            cv2.namedWindow("Preprocessor Comparison", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Preprocessor Comparison", scaled_grid)

            # Use non-blocking input check with cv2 event processing
            # NOTE: cv2.waitKey() is REQUIRED here - OpenCV won't render the image without it.
            # We use select() to check for Enter key in terminal without blocking.
            import select
            import sys
            while True:
                cv2.imshow("Preprocessor Comparison", scaled_grid)
                cv2.waitKey(100)  # Required for OpenCV to process GUI events and render
                # Check for terminal input (non-blocking)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()  # Consume the input
                    break
            cv2.destroyAllWindows()

        # Show best result
        best_method, best_facelets, best_classifications, best_conf = sorted_results[0]
        print(f"\nBest preprocessing method: {best_method} ({best_conf:.1f}% avg confidence)")
        print_classification_results(best_classifications, f"Best ({best_method})")

    finally:
        with suppress_output():
            camera.close()


def camera_classifier_preprocess_comparison_mode(display=True, use_v2: bool = False, use_v3: bool = False,
                                                  use_v4: bool = False, use_v5: bool = False, use_auto: bool = False,
                                                  rotate: bool = False):
    """
    Mode 6: Capture a single face and compare all preprocessing methods for color classification.

    Displays a tiled grid showing the classification result for each preprocessing
    method applied to the facelets, with the method name overlaid on each tile.

    Args:
        display: If True, show the comparison grid on display
        use_v2: If True, use v2 segmenter
        use_v3: If True, use v3 segmenter
        use_v4: If True, use v4 segmenter
        use_v5: If True, use v5 segmenter
        use_auto: If True, use auto-selecting segmenter
        rotate: If True, rotate captured images 180 degrees
    """
    if not JETSON_AVAILABLE:
        print("\nError: Camera mode requires Jetson hardware with IMX219 camera.")
        return

    print("\n" + "=" * 50)
    print("  CLASSIFIER PREPROCESSOR COMPARISON MODE")
    print("=" * 50)
    print("\nThis mode captures one face, segments it once, then shows")
    print("classification results for all preprocessing methods side-by-side.")

    # Initialize segmenter
    if use_auto:
        segmenter_name = "FaceletSegmenterAuto"
        segmenter_class = FaceletSegmenterAuto
    elif use_v5:
        segmenter_name = "FaceletSegmenterV5"
        segmenter_class = FaceletSegmenterV5
    elif use_v4:
        segmenter_name = "FaceletSegmenterV4"
        segmenter_class = FaceletSegmenterV4
    elif use_v3:
        segmenter_name = "FaceletSegmenterV3"
        segmenter_class = FaceletSegmenterV3
    elif use_v2:
        segmenter_name = "FaceletSegmenterV2"
        segmenter_class = FaceletSegmenterV2
    else:
        segmenter_name = "FaceletSegmenter"
        segmenter_class = FaceletSegmenter

    debug_print(f"Initializing {segmenter_name}...")
    start_time = time.time()
    with suppress_output():
        segmenter = segmenter_class()
    segmenter_time = time.time() - start_time
    debug_print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    debug_print("Initializing FaceletColorClassifier...")
    start_time = time.time()
    with suppress_output():
        classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    debug_print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    debug_print("Initializing ImagePreprocessor...")
    preprocessor = ImagePreprocessor()
    preprocess_methods = preprocessor.get_available_methods()
    debug_print(f"Found {len(preprocess_methods)} preprocessing methods")

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

        # Capture with live preview
        image = camera.capture_with_preview(display=display, rotate=rotate)

        if image is None:
            print("Error: Failed to capture image.")
            return

        print(f"\nCaptured image: {image.shape[1]}x{image.shape[0]}")

        # Segment once (no preprocessing for segmentation)
        print("\nSegmenting face...")
        facelets = segmenter.segment(image)

        if facelets is None:
            print("Error: Failed to segment face. Could not detect cube.")
            return

        print(f"Segmented into {facelets.shape[0]}x{facelets.shape[1]} facelets")
        print(f"\nProcessing {len(preprocess_methods)} preprocessing methods for classification...")

        # Process each preprocessing method on the facelets
        results = []  # List of (method_name, preprocessed_facelets, classifications, avg_conf)

        for i, method in enumerate(preprocess_methods):
            print(f"  [{i+1}/{len(preprocess_methods)}] {method}...", end=" ", flush=True)

            # Apply preprocessing to each facelet
            if method.lower() != 'none':
                preprocessed_facelets = np.zeros_like(facelets)
                for row in range(3):
                    for col in range(3):
                        preprocessed_facelets[row, col] = preprocessor.apply(method, facelets[row, col])
            else:
                preprocessed_facelets = facelets.copy()

            # Classify
            classifications = classifier.classify_face(preprocessed_facelets)

            # Calculate average confidence
            total_conf = 0
            for row in range(3):
                for col in range(3):
                    _, conf = classifications[row, col]
                    total_conf += conf
            avg_conf = total_conf / 9

            results.append((method, preprocessed_facelets, classifications, avg_conf))
            print(f"{avg_conf:.1f}%")

        # Sort by confidence for ranking
        sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

        # Print ranking
        print("\n" + "=" * 50)
        print("CLASSIFIER PREPROCESSING COMPARISON RESULTS")
        print("=" * 50)
        print(f"\n{'Rank':<5} {'Method':<20} {'Avg Confidence':<15}")
        print("-" * 40)
        for rank, (method, _, _, avg_conf) in enumerate(sorted_results, 1):
            print(f"{rank:<5} {method:<20} {avg_conf:>10.1f}%")

        # Create comparison grid image
        # Layout: 5 columns (fits 22 methods in 5 rows)
        cols = 5
        rows = (len(results) + cols - 1) // cols

        # Each tile: 3x3 facelet images + 3x3 color grid + label
        tile_w = 200
        tile_h = 200
        facelet_display_size = 20  # Size for each facelet image in the 3x3 grid
        color_square_size = 40

        grid_w = cols * tile_w
        grid_h = rows * tile_h + 40  # Extra space for title

        comparison_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        comparison_grid[:] = (40, 40, 40)  # Dark gray background

        # Title
        cv2.putText(comparison_grid, f"Classifier Preprocessing Comparison ({segmenter_name})",
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Color map for display
        COLOR_BGR = {
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0)
        }

        for idx, (method, preprocessed_facelets, classifications, avg_conf) in enumerate(results):
            row = idx // cols
            col = idx % cols

            x_base = col * tile_w
            y_base = row * tile_h + 40  # Offset for title

            # Draw 3x3 preprocessed facelet images (small thumbnails)
            facelet_grid_size = 3 * facelet_display_size
            facelet_start_x = x_base + (tile_w - facelet_grid_size) // 2
            facelet_start_y = y_base + 5

            for r in range(3):
                for c in range(3):
                    facelet_img = preprocessed_facelets[r, c]
                    facelet_thumb = cv2.resize(facelet_img, (facelet_display_size, facelet_display_size))
                    fx = facelet_start_x + c * facelet_display_size
                    fy = facelet_start_y + r * facelet_display_size
                    comparison_grid[fy:fy+facelet_display_size, fx:fx+facelet_display_size] = facelet_thumb

            # Draw 3x3 color grid below facelet images
            color_grid_size = 3 * color_square_size
            grid_start_x = x_base + (tile_w - color_grid_size) // 2
            grid_start_y = facelet_start_y + facelet_grid_size + 5

            for r in range(3):
                for c in range(3):
                    color_name, conf = classifications[r, c]
                    bgr = COLOR_BGR.get(color_name, (128, 128, 128))

                    fx = grid_start_x + c * color_square_size
                    fy = grid_start_y + r * color_square_size

                    # Fill color square
                    cv2.rectangle(comparison_grid, (fx+1, fy+1),
                                 (fx+color_square_size-1, fy+color_square_size-1), bgr, -1)
                    # Border
                    cv2.rectangle(comparison_grid, (fx, fy),
                                 (fx+color_square_size, fy+color_square_size), (80, 80, 80), 1)

            # Method name and confidence label at bottom
            label = f"{method}"
            conf_label = f"{avg_conf:.1f}%"

            # Find rank for color coding
            rank = next(i for i, (m, _, _, _) in enumerate(sorted_results, 1) if m == method)
            if rank <= 3:
                label_color = (0, 255, 0)  # Green for top 3
            elif rank <= 6:
                label_color = (0, 255, 255)  # Yellow for top 6
            else:
                label_color = (200, 200, 200)  # Gray for rest

            label_y = grid_start_y + 3 * color_square_size + 15
            cv2.putText(comparison_grid, label, (x_base + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)
            cv2.putText(comparison_grid, conf_label, (x_base + tile_w - 45, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)

        # Save the comparison grid
        output_path = os.path.join("output_facelets", "classifier_preprocess_comparison.jpg")
        os.makedirs("output_facelets", exist_ok=True)
        cv2.imwrite(output_path, comparison_grid)
        print(f"\nSaved comparison grid to {output_path}")

        # Display if requested
        if display:
            # Get screen dimensions and scale grid to fill the screen
            try:
                # Try to get screen size from X display
                import subprocess
                result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'dimensions:' in line:
                        dims = line.split()[1].split('x')
                        screen_w, screen_h = int(dims[0]), int(dims[1])
                        break
                else:
                    # Fallback to common Jetson display resolution
                    screen_w, screen_h = 1920, 1080
            except Exception:
                # Fallback to common Jetson display resolution
                screen_w, screen_h = 1920, 1080

            # Calculate scale factor to fill screen while maintaining aspect ratio
            scale_w = screen_w / grid_w
            scale_h = screen_h / grid_h
            scale = min(scale_w, scale_h) * 0.95  # 95% of screen for some margin

            # Scale the comparison grid
            new_w = int(grid_w * scale)
            new_h = int(grid_h * scale)
            scaled_grid = cv2.resize(comparison_grid, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            print(f"\nDisplaying at {new_w}x{new_h} (scaled from {grid_w}x{grid_h})")
            print("Press Enter to close the display...")

            cv2.namedWindow("Classifier Preprocess Comparison", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Classifier Preprocess Comparison", scaled_grid)

            # Use non-blocking input check with cv2 event processing
            # NOTE: cv2.waitKey() is REQUIRED here - OpenCV won't render the image without it.
            # We use select() to check for Enter key in terminal without blocking.
            import select
            import sys
            while True:
                cv2.imshow("Classifier Preprocess Comparison", scaled_grid)
                cv2.waitKey(100)  # Required for OpenCV to process GUI events and render
                # Check for terminal input (non-blocking)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()  # Consume the input
                    break
            cv2.destroyAllWindows()

        # Show best result
        best_method, best_facelets, best_classifications, best_conf = sorted_results[0]
        print(f"\nBest preprocessing method: {best_method} ({best_conf:.1f}% avg confidence)")
        print_classification_results(best_classifications, f"Best ({best_method})")

    finally:
        with suppress_output():
            camera.close()


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
        '--v2',
        action='store_true',
        help='Use v2 segmenter with improved detection (contour-based, perspective correction)'
    )
    parser.add_argument(
        '--v3',
        action='store_true',
        help='Use v3 segmenter with contour-based facelet detection'
    )
    parser.add_argument(
        '--v4',
        action='store_true',
        help='Use v4 segmenter with OpenCV square detection (Canny + contours)'
    )
    parser.add_argument(
        '--v5',
        action='store_true',
        help='Use v5 segmenter with brightness-based Otsu thresholding (Greg\'s CV)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Use auto-selecting segmenter (analyzes image to choose best algorithm)'
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
    args = parser.parse_args()

    # Validate preprocessing options
    preprocessor = ImagePreprocessor()
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

    # Show Jetson status
    if JETSON_AVAILABLE:
        print("\n[Jetson detected - Camera modes available]")
        if args.display:
            print("[Display mode enabled - Images will be shown on monitor]")
    else:
        print("\n[Running on non-Jetson platform - File modes only]")

    # Show segmenter version
    if args.auto:
        print("[Using AUTO segmenter - automatically selects best algorithm per image]")
    elif args.v5:
        print("[Using V5 segmenter - brightness-based Otsu thresholding (Greg's CV)]")
    elif args.v4:
        print("[Using V4 segmenter - OpenCV square detection (Canny + contours)]")
    elif args.v3:
        print("[Using V3 segmenter - contour-based facelet detection]")
    elif args.v2:
        print("[Using V2 segmenter - improved detection with perspective correction]")
    else:
        print("[Using V1 segmenter - standard detection]")

    if args.rotate:
        print("[Camera rotation enabled - images will be rotated 180 degrees]")

    # Show preprocessing options if set
    if args.segmenter_preprocess:
        print(f"[Segmenter preprocessing: {args.segmenter_preprocess}]")
    if args.cc_preprocess:
        print(f"[Color classifier preprocessing: {args.cc_preprocess}]")

    # Menu-driven loop
    while True:
        print("\n" + "-" * 50)
        print("Select Mode:")
        print("  1. Single Face (File)  - Load image and classify colors")
        print("  2. Full Cube (File)    - Load 6 images and solve cube")
        if JETSON_AVAILABLE:
            print("  3. Single Face (Camera) - Capture and classify one face")
            print("  4. Full Cube (Camera)   - Capture 6 faces and solve")
            print("  5. Segmenter Preprocess - Compare all preprocessors for segmentation")
            print("  6. Classifier Preprocess - Compare all preprocessors for color classification")
        print("  q. Quit")
        print("-" * 50)

        choice = input("> ").strip().lower()

        if choice == '1':
            single_face_mode(use_v2=args.v2, use_v3=args.v3, use_v4=args.v4, use_v5=args.v5, use_auto=args.auto,
                            segmenter_preprocess=args.segmenter_preprocess, cc_preprocess=args.cc_preprocess)
        elif choice == '2':
            full_cube_mode(use_v2=args.v2, use_v3=args.v3, use_v4=args.v4, use_v5=args.v5, use_auto=args.auto, display=args.display,
                          segmenter_preprocess=args.segmenter_preprocess, cc_preprocess=args.cc_preprocess)
        elif choice == '3' and JETSON_AVAILABLE:
            camera_single_face_mode(display=args.display, use_v2=args.v2, use_v3=args.v3, use_v4=args.v4, use_v5=args.v5, use_auto=args.auto, rotate=args.rotate,
                                   segmenter_preprocess=args.segmenter_preprocess, cc_preprocess=args.cc_preprocess)
        elif choice == '4' and JETSON_AVAILABLE:
            camera_full_cube_mode(display=args.display, use_v2=args.v2, use_v3=args.v3, use_v4=args.v4, use_v5=args.v5, use_auto=args.auto, rotate=args.rotate,
                                 segmenter_preprocess=args.segmenter_preprocess, cc_preprocess=args.cc_preprocess)
        elif choice == '5' and JETSON_AVAILABLE:
            camera_segmenter_preprocess_comparison_mode(display=args.display, use_v2=args.v2, use_v3=args.v3, use_v4=args.v4, use_v5=args.v5, use_auto=args.auto, rotate=args.rotate)
        elif choice == '6' and JETSON_AVAILABLE:
            camera_classifier_preprocess_comparison_mode(display=args.display, use_v2=args.v2, use_v3=args.v3, use_v4=args.v4, use_v5=args.v5, use_auto=args.auto, rotate=args.rotate)
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            valid_choices = "1, 2, 3, 4, 5, 6, or q" if JETSON_AVAILABLE else "1, 2, or q"
            print(f"Invalid choice. Please enter {valid_choices}.")


if __name__ == "__main__":
    main()
