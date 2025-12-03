#!/usr/bin/env python3
"""
Display a logged face and its facelets based on serial number.

Usage:
    python tools/display_log.py <serial>
    python tools/display_log.py 00000
    python tools/display_log.py 0       # Will be zero-padded to 00000
"""

import cv2
import sys
import os
import numpy as np
import select

LOG_DIR = "log"


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/display_log.py <serial>")
        print("Example: python tools/display_log.py 00000")
        sys.exit(1)

    # Parse serial number and zero-pad to 5 digits
    try:
        serial = int(sys.argv[1])
        serial_str = f"{serial:05d}"
    except ValueError:
        serial_str = sys.argv[1]

    print(f"Looking for files with serial: {serial_str}")

    # Find the face file
    face_files = []
    for f in os.listdir(LOG_DIR):
        if f.endswith(f"_{serial_str}.jpg") and "_facelet_" not in f:
            face_files.append(f)

    if not face_files:
        print(f"No face files found with serial {serial_str}")
        sys.exit(1)

    # Process each face found
    for face_file in face_files:
        # Extract side name: (side)_(serial).jpg
        side_name = face_file.rsplit('_', 1)[0]
        print(f"\nProcessing side: {side_name}")

        # Load face image
        face_path = os.path.join(LOG_DIR, face_file)
        face_img = cv2.imread(face_path)
        if face_img is None:
            print(f"  Error loading face: {face_path}")
            continue
        print(f"  Face loaded: {face_img.shape}")

        # Load all 9 facelets, add white border to each, then arrange in grid
        cell_size = 100  # Size of each facelet display
        border = 2       # White border around each facelet
        spacing = 2      # Space between facelets (same as border)

        bordered_size = cell_size + border * 2  # Facelet + border on all sides
        grid_size = bordered_size * 3 + spacing * 2  # 3 bordered cells + 2 gaps

        # Black background
        facelet_grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        for row in range(3):
            for col in range(3):
                facelet_file = f"{side_name}_facelet_{row}_{col}_{serial_str}.jpg"
                facelet_path = os.path.join(LOG_DIR, facelet_file)

                if os.path.exists(facelet_path):
                    img = cv2.imread(facelet_path)
                    if img is not None:
                        # Resize facelet to cell size
                        img = cv2.resize(img, (cell_size, cell_size))
                        # Add white border around facelet
                        bordered = cv2.copyMakeBorder(img, border, border, border, border,
                                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
                        # Place in grid
                        y1 = row * (bordered_size + spacing)
                        x1 = col * (bordered_size + spacing)
                        facelet_grid[y1:y1+bordered_size, x1:x1+bordered_size] = bordered
                        print(f"  Facelet [{row}][{col}] loaded -> grid[{y1}:{y1+bordered_size}, {x1}:{x1+bordered_size}]")
                    else:
                        print(f"  Facelet [{row}][{col}] failed to load")
                else:
                    print(f"  Facelet [{row}][{col}] not found: {facelet_file}")

        # Scale face to match facelet grid height
        scale = grid_size / face_img.shape[0]
        new_width = int(face_img.shape[1] * scale)
        face_img = cv2.resize(face_img, (new_width, grid_size))

        # Combine face and facelet grid side by side
        gap = spacing
        gap_img = np.zeros((grid_size, gap, 3), dtype=np.uint8)
        combined = np.hstack([face_img, gap_img, facelet_grid])

        # Display
        window_name = f"Log {serial_str} - {side_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, combined)

    print("\nPress Enter to close...")

    # Keep updating display while waiting for terminal input
    while True:
        cv2.waitKey(30)
        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
