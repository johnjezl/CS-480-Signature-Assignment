"""
Rubik's Cube Face Scanner - Main Application

Menu-driven application with multiple modes:
1. Single Face Mode: Scan one image file and classify colors
2. Full Cube Mode: Scan all 6 faces from files and solve the cube
3. Camera Single Face: Capture one face from camera and classify (Jetson only)
4. Camera Full Cube: Capture all 6 faces from camera and solve (Jetson only)

Usage:
    python main.py [--display] [--v2]

Options:
    --display    Show captured images on display (for Jetson with monitor)
    --v2         Use v2 segmenter with improved detection algorithms
"""

import cv2
import numpy as np
import os
import json
import time
import argparse

from facelet_segmenter import FaceletSegmenter
from facelet_segmenter_v2 import FaceletSegmenterV2
from FaceletColorClassifier import FaceletColorClassifier
from IDASolver import IDASolver

# Try to import Jetson camera module
try:
    from JetsonCamera import JetsonCamera, is_jetson, display_image, display_images_grid
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


# Color abbreviation map
COLOR_ABBREV = {
    'white': 'W', 'yellow': 'Y', 'red': 'R',
    'orange': 'O', 'blue': 'B', 'green': 'G'
}

# Face names for the solver (in order of input)
FACE_NAMES = ['up', 'down', 'front', 'back', 'left', 'right']
FACE_DISPLAY_NAMES = ['Up (Yellow)', 'Down (White)', 'Front (Blue)',
                      'Back (Green)', 'Left (Orange)', 'Right (Red)']

# Log directory for captured images
LOG_DIR = "log"


def get_next_serial():
    """
    Get the next available 5-digit serial number by scanning the log directory.

    Returns:
        int: Next serial number to use
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    # Find existing files and get the highest serial number
    max_serial = -1
    for filename in os.listdir(LOG_DIR):
        if filename.endswith('.jpg'):
            # Extract serial number from filename (last 5 digits before .jpg)
            try:
                # Format: (side)_(serial).jpg or (side)_facelet_x_y_(serial).jpg
                base = filename[:-4]  # Remove .jpg
                serial_str = base.split('_')[-1]
                serial = int(serial_str)
                if serial > max_serial:
                    max_serial = serial
            except (ValueError, IndexError):
                continue

    return max_serial + 1


def format_serial(serial):
    """Format serial number as 5-digit string with leading zeros."""
    return f"{serial:05d}"


def print_classification_results(classifications, face_name=None):
    """
    Print the classification results in a formatted grid.

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

    # Print visual grid
    print("\nFace Layout:")
    print("-" * 28)
    for row in range(3):
        row_str = "|"
        for col in range(3):
            color, conf = classifications[row, col]
            row_str += f" {color:^6s} |"
        print(row_str)
        print("-" * 28)

    # Print detailed results with confidence
    print("\nDetailed Results:")
    for row in range(3):
        for col in range(3):
            color, conf = classifications[row, col]
            print(f"  [{row},{col}]: {color:8s} ({conf:5.1f}%)")

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


def log_face_and_facelets(image, facelets, side_name, serial):
    """
    Log the face image and extracted facelets to the log directory.

    Args:
        image: BGR numpy array of the full face image
        facelets: numpy array of shape (3, 3, 64, 64, 3)
        side_name: Name of the cube side (up, down, front, back, left, right)
        serial: Serial number for this capture
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    serial_str = format_serial(serial)

    # Save the face image: (side)_(serial).jpg
    face_filename = f"{side_name}_{serial_str}.jpg"
    face_path = os.path.join(LOG_DIR, face_filename)
    cv2.imwrite(face_path, image)

    # Save each facelet: (side)_facelet_x_y_(serial).jpg
    for row in range(3):
        for col in range(3):
            facelet = facelets[row, col]
            facelet_filename = f"{side_name}_facelet_{row}_{col}_{serial_str}.jpg"
            facelet_path = os.path.join(LOG_DIR, facelet_filename)
            cv2.imwrite(facelet_path, facelet)

    print(f"Logged face and 9 facelets to {LOG_DIR}/ (serial: {serial_str})")


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

    # Display
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, combined)
    cv2.waitKey(30)  # Process window events


def process_single_face(image_path, segmenter, classifier, side_name=None, display=False):
    """
    Process a single face image from file: segment and classify.

    Args:
        image_path: Path to the image file
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        side_name: Name of the cube side for logging
        display: If True, show face and facelets on display

    Returns:
        classifications or None on error
    """
    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    return process_image(image, segmenter, classifier, side_name, display)


def process_image(image, segmenter, classifier, side_name=None, display=False):
    """
    Process an image (BGR numpy array): segment and classify.

    Args:
        image: BGR numpy array (height, width, 3)
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        side_name: Name of the cube side for logging (up, down, front, back, left, right)
        display: If True, show face and facelets on display

    Returns:
        classifications or None on error
    """
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Segment the image into facelets
    print("Segmenting image into 3x3 facelets...")
    start_time = time.time()
    facelets = segmenter.segment(image)
    segment_time = time.time() - start_time
    print(f"Facelets shape: {facelets.shape} (took {segment_time:.3f}s)")

    # Display face and facelets if requested
    if display:
        window_name = f"Segmented: {side_name}" if side_name else "Segmented Face"
        display_face_and_facelets(image, facelets, window_name)

    # Log face and facelets with serial number
    if side_name:
        serial = get_next_serial()
        log_face_and_facelets(image, facelets, side_name, serial)

    # Classify the colors
    print("Classifying facelet colors...")
    start_time = time.time()
    classifications = classifier.classify_face(facelets)
    classify_time = time.time() - start_time
    print(f"Classification complete (took {classify_time:.3f}s)")

    return classifications


def single_face_mode(use_v2: bool = False):
    """Mode 1: Process a single face image."""
    print("\n" + "=" * 50)
    print("  SINGLE FACE MODE")
    print("=" * 50)

    # Initialize components with timing
    segmenter_name = "FaceletSegmenterV2" if use_v2 else "FaceletSegmenter"
    print(f"\nInitializing {segmenter_name}...")
    start_time = time.time()
    segmenter = FaceletSegmenterV2() if use_v2 else FaceletSegmenter()
    segmenter_time = time.time() - start_time
    print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    print("\nInitializing FaceletColorClassifier...")
    start_time = time.time()
    classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Get image path
    image_path = get_image_path()
    if image_path is None:
        print("Cancelled.")
        return

    # Process the face
    classifications = process_single_face(image_path, segmenter, classifier, side_name="single")
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


def full_cube_mode(use_v2: bool = False):
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
    segmenter_name = "FaceletSegmenterV2" if use_v2 else "FaceletSegmenter"
    print(f"\nInitializing {segmenter_name}...")
    start_time = time.time()
    segmenter = FaceletSegmenterV2() if use_v2 else FaceletSegmenter()
    segmenter_time = time.time() - start_time
    print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    print("\nInitializing FaceletColorClassifier...")
    start_time = time.time()
    classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    # Dictionary to store face data for the solver
    cube_data = {}

    # Process each face
    for i, (face_key, face_display) in enumerate(zip(FACE_NAMES, FACE_DISPLAY_NAMES)):
        print(f"\n{'#' * 50}")
        print(f"  FACE {i+1}/6: {face_display}")
        print("#" * 50)

        image_path = face_files[face_key]

        # Process the face with display enabled
        classifications = process_single_face(image_path, segmenter, classifier, side_name=face_key, display=True)
        if classifications is None:
            print("\nError processing face. Aborting cube solve.")
            return

        # Get results
        face_string, face_array = print_classification_results(classifications, face_display)

        # Store for solver
        cube_data[face_key] = face_array

        print(f"\n{face_display} face captured successfully!")

        # Pause to review the displayed image
        remaining = 6 - (i + 1)
        if remaining > 0:
            print(f"\n{remaining} face(s) remaining. Press Enter to continue (or 'q' to cancel)...")
        else:
            print("\nPress Enter to continue to solver (or 'q' to cancel)...")
        user_input = input("> ").strip().lower()
        cv2.destroyAllWindows()
        if user_input == 'q':
            print("\nCancelled. Aborting cube solve.")
            return

    # Close any remaining display windows
    cv2.destroyAllWindows()

    # All faces captured - prepare solver input
    print("\n" + "=" * 50)
    print("  ALL FACES CAPTURED")
    print("=" * 50)

    # Display summary
    print("\nCube Configuration:")
    for face_key, face_display in zip(FACE_NAMES, FACE_DISPLAY_NAMES):
        face_str = ''.join(cube_data[face_key])
        print(f"  {face_display:20s}: {face_str}")

    # Write JSON file for solver
    solver_input = {"cube": cube_data}
    json_path = "AStar_in.json"
    print(f"\nWriting solver input to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(solver_input, f, indent=2)
    print("Done!")

    # Call the solver
    print("\n" + "=" * 50)
    print("  SOLVING CUBE")
    print("=" * 50)

    print("\nInitializing IDASolver...")
    start_time = time.time()
    solver = IDASolver()
    solver_init_time = time.time() - start_time
    print(f"IDASolver ready (took {solver_init_time:.3f}s)")

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


def camera_single_face_mode(display=False, use_v2: bool = False):
    """
    Mode 3: Capture a single face from camera and classify.

    Args:
        display: If True, show captured images on display
        use_v2: If True, use v2 segmenter with improved detection
    """
    if not JETSON_AVAILABLE:
        print("\nError: Camera mode requires Jetson hardware with IMX219 camera.")
        return

    print("\n" + "=" * 50)
    print("  CAMERA SINGLE FACE MODE")
    print("=" * 50)

    # Initialize components with timing
    segmenter_name = "FaceletSegmenterV2" if use_v2 else "FaceletSegmenter"
    print(f"\nInitializing {segmenter_name}...")
    start_time = time.time()
    segmenter = FaceletSegmenterV2() if use_v2 else FaceletSegmenter()
    segmenter_time = time.time() - start_time
    print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    print("\nInitializing FaceletColorClassifier...")
    start_time = time.time()
    classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    print("\nInitializing JetsonCamera...")
    start_time = time.time()
    camera = JetsonCamera()
    if not camera.open():
        print("Error: Failed to open camera.")
        return
    camera_time = time.time() - start_time
    print(f"JetsonCamera ready (took {camera_time:.3f}s)")

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
        image = camera.capture_with_preview(display=display)

        if image is None:
            print("Error: Failed to capture image.")
            return

        # Save the captured image
        capture_path = os.path.join("output_facelets", "camera_capture.jpg")
        os.makedirs("output_facelets", exist_ok=True)
        cv2.imwrite(capture_path, image)
        print(f"Saved captured image to {capture_path}")

        # Process the image
        print("\nProcessing captured image...")
        classifications = process_image(image, segmenter, classifier, side_name="single", display=display)

        if classifications is None:
            return

        # Output results
        face_string, _ = print_classification_results(classifications)
        print(f"\nScan complete! Face string: {face_string}")

        # Close display window if it was opened
        if display:
            input("\nPress Enter to close the image display...")
            cv2.destroyAllWindows()

    finally:
        camera.close()


def camera_full_cube_mode(display=False, use_v2: bool = False):
    """
    Mode 4: Capture all 6 faces from camera and solve the cube.

    Args:
        display: If True, show captured images on display
        use_v2: If True, use v2 segmenter with improved detection
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
    segmenter_name = "FaceletSegmenterV2" if use_v2 else "FaceletSegmenter"
    print(f"\nInitializing {segmenter_name}...")
    start_time = time.time()
    segmenter = FaceletSegmenterV2() if use_v2 else FaceletSegmenter()
    segmenter_time = time.time() - start_time
    print(f"{segmenter_name} ready (took {segmenter_time:.3f}s)")

    print("\nInitializing FaceletColorClassifier...")
    start_time = time.time()
    classifier = FaceletColorClassifier()
    classifier_time = time.time() - start_time
    print(f"FaceletColorClassifier ready (took {classifier_time:.3f}s)")

    print("\nInitializing JetsonCamera...")
    start_time = time.time()
    camera = JetsonCamera()
    if not camera.open():
        print("Error: Failed to open camera.")
        return
    camera_time = time.time() - start_time
    print(f"JetsonCamera ready (took {camera_time:.3f}s)")

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

    try:
        # Process each face
        for i, (face_key, face_display) in enumerate(zip(FACE_NAMES, FACE_DISPLAY_NAMES)):
            print(f"\n{'#' * 50}")
            print(f"  FACE {i+1}/6: {face_display}")
            print("#" * 50)

            # Show orientation instructions
            print(f"\nOrientation: {FACE_INSTRUCTIONS[face_key]}")
            print("\nPress Enter when ready, or 'q' to cancel...")

            user_input = input("> ").strip().lower()
            if user_input == 'q':
                print("\nCancelled. Aborting cube solve.")
                return

            # Capture with live preview
            image = camera.capture_with_preview(display=display)

            if image is None:
                print("Error: Failed to capture image. Aborting.")
                return

            # Store for later display
            captured_images.append(image.copy())

            # Process the image
            print("\nProcessing captured image...")
            classifications = process_image(image, segmenter, classifier, side_name=face_key, display=display)

            if classifications is None:
                print("\nError processing face. Aborting cube solve.")
                return

            # Get results
            face_string, face_array = print_classification_results(classifications, face_display)

            # Store for solver
            cube_data[face_key] = face_array

            print(f"\n{face_display} face captured successfully!")

            # Show progress
            remaining = 6 - (i + 1)
            if remaining > 0:
                print(f"\n{remaining} face(s) remaining...")

    finally:
        camera.close()

    # All faces captured - display all images if requested
    if display and captured_images:
        print("\nDisplaying all captured faces...")
        display_images_grid(captured_images, labels=FACE_DISPLAY_NAMES,
                            window_name="All Cube Faces", cols=3)

    # Prepare solver input
    print("\n" + "=" * 50)
    print("  ALL FACES CAPTURED")
    print("=" * 50)

    # Display summary
    print("\nCube Configuration:")
    for face_key, face_display in zip(FACE_NAMES, FACE_DISPLAY_NAMES):
        face_str = ''.join(cube_data[face_key])
        print(f"  {face_display:20s}: {face_str}")

    # Write JSON file for solver
    solver_input = {"cube": cube_data}
    json_path = "AStar_in.json"
    print(f"\nWriting solver input to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(solver_input, f, indent=2)
    print("Done!")

    # Call the solver
    print("\n" + "=" * 50)
    print("  SOLVING CUBE")
    print("=" * 50)

    print("\nInitializing IDASolver...")
    start_time = time.time()
    solver = IDASolver()
    solver_init_time = time.time() - start_time
    print(f"IDASolver ready (took {solver_init_time:.3f}s)")

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
    args = parser.parse_args()

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
    if args.v2:
        print("[Using V2 segmenter - improved detection with perspective correction]")
    else:
        print("[Using V1 segmenter - standard detection]")

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
            single_face_mode(use_v2=args.v2)
        elif choice == '2':
            full_cube_mode(use_v2=args.v2)
        elif choice == '3' and JETSON_AVAILABLE:
            camera_single_face_mode(display=args.display, use_v2=args.v2)
        elif choice == '4' and JETSON_AVAILABLE:
            camera_full_cube_mode(display=args.display, use_v2=args.v2)
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            valid_choices = "1, 2, 3, 4, or q" if JETSON_AVAILABLE else "1, 2, or q"
            print(f"Invalid choice. Please enter {valid_choices}.")


if __name__ == "__main__":
    main()
