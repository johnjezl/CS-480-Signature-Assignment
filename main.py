"""
Rubik's Cube Face Scanner - Main Application

Menu-driven application with two modes:
1. Single Face Mode: Scan one image and classify colors
2. Full Cube Mode: Scan all 6 faces and solve the cube

Usage:
    python main.py
"""

import cv2
import numpy as np
import os
import json
import time

from facelet_segmenter import FaceletSegmenter
from FaceletColorClassifier import FaceletColorClassifier
from IDASolver import IDASolver


# Color abbreviation map
COLOR_ABBREV = {
    'white': 'W', 'yellow': 'Y', 'red': 'R',
    'orange': 'O', 'blue': 'B', 'green': 'G'
}

# Face names for the solver (in order of input)
FACE_NAMES = ['up', 'down', 'front', 'back', 'left', 'right']
FACE_DISPLAY_NAMES = ['Up (Yellow)', 'Down (White)', 'Front (Blue)',
                      'Back (Green)', 'Left (Orange)', 'Right (Red)']


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


def process_single_face(image_path, segmenter, classifier):
    """
    Process a single face image: segment and classify.

    Args:
        image_path: Path to the image file
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance

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

    # Segment the image into facelets
    print("Segmenting image into 3x3 facelets...")
    start_time = time.time()
    facelets = segmenter.segment(image)
    segment_time = time.time() - start_time
    print(f"Facelets shape: {facelets.shape} (took {segment_time:.3f}s)")

    # Classify the colors
    print("Classifying facelet colors...")
    start_time = time.time()
    classifications = classifier.classify_face(facelets)
    classify_time = time.time() - start_time
    print(f"Classification complete (took {classify_time:.3f}s)")

    return classifications


def single_face_mode():
    """
    Mode 1: Process a single face image.
    """
    print("\n" + "=" * 50)
    print("  SINGLE FACE MODE")
    print("=" * 50)

    # Initialize components with timing
    print("\nInitializing FaceletSegmenter...")
    start_time = time.time()
    segmenter = FaceletSegmenter()
    segmenter_time = time.time() - start_time
    print(f"FaceletSegmenter ready (took {segmenter_time:.3f}s)")

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
    classifications = process_single_face(image_path, segmenter, classifier)
    if classifications is None:
        return

    # Output results
    face_string, _ = print_classification_results(classifications)
    print(f"\nScan complete! Face string: {face_string}")


def full_cube_mode():
    """
    Mode 2: Process all 6 faces and solve the cube.
    """
    print("\n" + "=" * 50)
    print("  FULL CUBE SOLVER MODE")
    print("=" * 50)
    print("\nYou will be prompted to enter images for all 6 faces of the cube.")
    print("Face order: Up (Yellow), Down (White), Front (Blue),")
    print("            Back (Green), Left (Orange), Right (Red)")
    print("\nMake sure to orient the cube correctly for each face!")

    # Initialize components with timing
    print("\nInitializing FaceletSegmenter...")
    start_time = time.time()
    segmenter = FaceletSegmenter()
    segmenter_time = time.time() - start_time
    print(f"FaceletSegmenter ready (took {segmenter_time:.3f}s)")

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

        # Get image path
        image_path = get_image_path(f"Enter image for {face_display} face:")
        if image_path is None:
            print("\nCancelled. Aborting cube solve.")
            return

        # Process the face
        classifications = process_single_face(image_path, segmenter, classifier)
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


def main():
    """Main entry point for the application."""
    print("=" * 50)
    print("  RUBIK'S CUBE SCANNER & SOLVER")
    print("  Segmentation + Color Classification + IDA* Solver")
    print("=" * 50)

    # Menu-driven loop
    while True:
        print("\n" + "-" * 50)
        print("Select Mode:")
        print("  1. Single Face - Scan one image and classify colors")
        print("  2. Full Cube   - Scan all 6 faces and solve the cube")
        print("  q. Quit")
        print("-" * 50)

        choice = input("> ").strip().lower()

        if choice == '1':
            single_face_mode()
        elif choice == '2':
            full_cube_mode()
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or q.")


if __name__ == "__main__":
    main()
