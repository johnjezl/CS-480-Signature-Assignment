"""
Analyze Real Facelet Palette Tool

Analyzes facelets in dataset/real_facelets to compute average BGR values
for the center region of each color's facelets.

Usage:
    python tools/analyze_facelet_palette.py
    python tools/analyze_facelet_palette.py --input dataset/real_facelets
    python tools/analyze_facelet_palette.py --center-size 10

Output:
    Prints palette dictionary in SyntheticFaceletGenerator format.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path


def analyze_facelet_center(image_path, center_size=10):
    """
    Extract the average BGR value from the center region of a facelet.

    Args:
        image_path: Path to facelet image
        center_size: Size of center region to sample (default 10x10 pixels)

    Returns:
        tuple: (B, G, R) average values, or None on error
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load {image_path}")
        return None

    h, w = img.shape[:2]

    # Calculate center region bounds
    cx, cy = w // 2, h // 2
    half = center_size // 2

    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    # Extract center region
    center = img[y1:y2, x1:x2]

    # Compute mean BGR
    mean_bgr = np.mean(center, axis=(0, 1))

    return tuple(mean_bgr)


def analyze_color_directory(color_dir, center_size=10):
    """
    Analyze all facelets in a color directory.

    Args:
        color_dir: Path to color directory
        center_size: Size of center region to sample

    Returns:
        dict: Statistics including mean, std, min, max BGR values
    """
    bgr_values = []

    # Get all image files
    extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    for filename in os.listdir(color_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            filepath = os.path.join(color_dir, filename)
            bgr = analyze_facelet_center(filepath, center_size)
            if bgr is not None:
                bgr_values.append(bgr)

    if not bgr_values:
        return None

    bgr_array = np.array(bgr_values)

    return {
        'count': len(bgr_values),
        'mean': np.mean(bgr_array, axis=0),
        'std': np.std(bgr_array, axis=0),
        'min': np.min(bgr_array, axis=0),
        'max': np.max(bgr_array, axis=0),
        'all_values': bgr_values
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze facelet colors to generate palette')
    parser.add_argument('--input', '-i', default='datasets/real_facelets',
                       help='Input directory containing color subdirectories')
    parser.add_argument('--center-size', '-c', type=int, default=10,
                       help='Size of center region to sample (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed statistics')
    args = parser.parse_args()

    # Resolve input directory
    input_dir = args.input
    if not os.path.isabs(input_dir):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, '..', input_dir)
        input_dir = os.path.normpath(input_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("FACELET PALETTE ANALYZER")
    print("=" * 60)
    print(f"\nInput directory: {input_dir}")
    print(f"Center sample size: {args.center_size}x{args.center_size} pixels")
    print()

    # Expected color directories
    colors = ['white', 'yellow', 'red', 'orange', 'blue', 'green']

    results = {}

    for color in colors:
        color_dir = os.path.join(input_dir, color)
        if not os.path.isdir(color_dir):
            print(f"  {color:8s}: [not found]")
            continue

        stats = analyze_color_directory(color_dir, args.center_size)
        if stats is None:
            print(f"  {color:8s}: [no images]")
            continue

        results[color] = stats
        mean_bgr = stats['mean']
        print(f"  {color:8s}: {stats['count']:3d} images -> "
              f"BGR ({mean_bgr[0]:6.1f}, {mean_bgr[1]:6.1f}, {mean_bgr[2]:6.1f})")

    if args.verbose:
        print("\n" + "-" * 60)
        print("DETAILED STATISTICS")
        print("-" * 60)
        for color, stats in results.items():
            print(f"\n{color.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean:  B={stats['mean'][0]:.1f}, G={stats['mean'][1]:.1f}, R={stats['mean'][2]:.1f}")
            print(f"  Std:   B={stats['std'][0]:.1f}, G={stats['std'][1]:.1f}, R={stats['std'][2]:.1f}")
            print(f"  Min:   B={stats['min'][0]:.1f}, G={stats['min'][1]:.1f}, R={stats['min'][2]:.1f}")
            print(f"  Max:   B={stats['max'][0]:.1f}, G={stats['max'][1]:.1f}, R={stats['max'][2]:.1f}")

    # Output in SyntheticFaceletGenerator format
    print("\n" + "=" * 60)
    print("PALETTE OUTPUT (SyntheticFaceletGenerator format)")
    print("=" * 60)
    print("\n# Copy this into SyntheticFaceletGenerator.py palettes dict:")
    print("'real_facelets': {")

    for color in colors:
        if color in results:
            mean = results[color]['mean']
            # Round to integers
            b, g, r = int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))
            print(f"    '{color}': ({b}, {g}, {r}),")
        else:
            print(f"    # '{color}': (?, ?, ?),  # No data")

    print("}")

    # Also output raw values for easy copying
    print("\n# Or as a single-line dict:")
    palette_parts = []
    for color in colors:
        if color in results:
            mean = results[color]['mean']
            b, g, r = int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))
            palette_parts.append(f"'{color}': ({b}, {g}, {r})")

    print("{" + ", ".join(palette_parts) + "}")

    print()


if __name__ == "__main__":
    main()
