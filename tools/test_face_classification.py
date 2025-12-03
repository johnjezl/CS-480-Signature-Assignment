#!/usr/bin/env python3
"""
Test script for face segmentation and color classification.
Tests all images in input_faces directory and outputs detected colors for verification.
"""

import os
import glob
import shutil
import cv2
from facelet_segmenter import FaceletSegmenter
from FaceletColorClassifier import FaceletColorClassifier


def test_image(image_path: str, segmenter: FaceletSegmenter, classifier: FaceletColorClassifier,
               debug_dir: str = None):
    """
    Test a single image: segment it and classify colors.

    Args:
        image_path: Path to the image file
        segmenter: FaceletSegmenter instance
        classifier: FaceletColorClassifier instance
        debug_dir: Optional directory to save extracted facelets for debugging
    """
    image_name = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"Image: {image_name}")
    print(f"{'='*60}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not load image")
        return

    print(f"  Dimensions: {image.shape[1]}x{image.shape[0]}")

    # Segment the face
    try:
        facelets = segmenter.segment(image)
        print(f"  Segmentation: SUCCESS")
    except Exception as e:
        print(f"  Segmentation: FAILED - {e}")
        return

    # Save facelets to debug directory if specified
    if debug_dir is not None:
        # Create subdirectory for this image (use image name without extension)
        image_base = os.path.splitext(image_name)[0]
        image_debug_dir = os.path.join(debug_dir, image_base)
        os.makedirs(image_debug_dir, exist_ok=True)

        # Save each facelet (facelets is 3x3 array of 64x64 images)
        for row in range(3):
            for col in range(3):
                facelet_idx = row * 3 + col
                facelet = facelets[row, col]
                facelet_path = os.path.join(image_debug_dir, f"facelet_{facelet_idx}.png")
                cv2.imwrite(facelet_path, facelet)
        print(f"  Debug facelets saved to: {image_debug_dir}")

    # Classify colors
    try:
        classifications = classifier.classify_face_batch(facelets)
        print(f"  Classification: SUCCESS")
    except Exception as e:
        print(f"  Classification: FAILED - {e}")
        return

    # Output color grid
    print(f"\n  Facelet Colors (with confidence):")
    print(f"  ┌────────────────┬────────────────┬────────────────┐")

    for row in range(3):
        row_str = "  │"
        for col in range(3):
            color, confidence = classifications[row, col]
            cell = f" {color:6s} {confidence:5.1f}% "
            row_str += cell + "│"
        print(row_str)
        if row < 2:
            print(f"  ├────────────────┼────────────────┼────────────────┤")

    print(f"  └────────────────┴────────────────┴────────────────┘")

    # Simple text grid for easy verification
    print(f"\n  Simple Grid:")
    for row in range(3):
        colors = [classifications[row, col][0] for col in range(3)]
        print(f"    {colors[0]:8s} {colors[1]:8s} {colors[2]:8s}")

    # Facelet numbering reference
    print(f"\n  Facelet Positions:")
    print(f"    0  1  2")
    print(f"    3  4  5")
    print(f"    6  7  8")


def main():
    print("="*60)
    print("Face Segmentation & Color Classification Test")
    print("="*60)

    # Initialize components
    print("\nInitializing...")
    segmenter = FaceletSegmenter(output_size=64)
    classifier = FaceletColorClassifier(model_path='models/best_model.pth')
    print(f"  Segmenter: Ready")
    print(f"  Classifier: Ready (device: {classifier.device})")

    # Create debug directory for extracted facelets
    debug_dir = "facelets_for_debug"
    if os.path.exists(debug_dir):
        # Clear existing debug directory
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    print(f"  Debug output: {debug_dir}/")

    # Find all images in input_faces directory
    input_dir = "input_faces"

    # Get all image files (jpg, jpeg, png)
    image_patterns = [
        os.path.join(input_dir, "*.jpg"),
        os.path.join(input_dir, "*.jpeg"),
        os.path.join(input_dir, "*.png"),
        os.path.join(input_dir, "*.JPG"),
        os.path.join(input_dir, "*.JPEG"),
        os.path.join(input_dir, "*.PNG"),
    ]

    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))

    # Sort for consistent ordering
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"\nNo images found in {input_dir}/")
        return

    print(f"\nFound {len(image_files)} images in {input_dir}/")

    # Test each image
    for image_path in image_files:
        test_image(image_path, segmenter, classifier, debug_dir)

    # Check for subdirectories (e.g., Black Background)
    subdirs = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        print(f"\n\n{'#'*60}")
        print(f"# Subdirectory: {subdir}")
        print(f"{'#'*60}")

        sub_image_files = []
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            sub_image_files.extend(glob.glob(os.path.join(subdir_path, pattern)))

        sub_image_files = sorted(set(sub_image_files))

        if sub_image_files:
            print(f"Found {len(sub_image_files)} images")
            # Create subdirectory in debug output (use "Black" for "Black Background")
            subdir_debug_name = "Black" if "black" in subdir.lower() else subdir
            subdir_debug_dir = os.path.join(debug_dir, subdir_debug_name)
            os.makedirs(subdir_debug_dir, exist_ok=True)
            for image_path in sub_image_files:
                test_image(image_path, segmenter, classifier, subdir_debug_dir)

    print(f"\n\n{'='*60}")
    print("Test Complete")
    print("="*60)


if __name__ == "__main__":
    main()
