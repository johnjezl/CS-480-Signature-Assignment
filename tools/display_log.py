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

        # Scale face to reasonable size
        target_height = 400
        scale = target_height / face_img.shape[0]
        new_width = int(face_img.shape[1] * scale)
        face_img = cv2.resize(face_img, (new_width, target_height))

        # Load all 9 facelets
        cell_size = target_height // 3
        facelet_grid = np.zeros((target_height, target_height, 3), dtype=np.uint8)

        for row in range(3):
            for col in range(3):
                facelet_file = f"{side_name}_facelet_{row}_{col}_{serial_str}.jpg"
                facelet_path = os.path.join(LOG_DIR, facelet_file)

                if os.path.exists(facelet_path):
                    img = cv2.imread(facelet_path)
                    if img is not None:
                        # Resize to cell size
                        img = cv2.resize(img, (cell_size, cell_size))
                        # Place in grid
                        y1 = row * cell_size
                        x1 = col * cell_size
                        facelet_grid[y1:y1+cell_size, x1:x1+cell_size] = img
                        print(f"  Facelet [{row}][{col}] loaded")
                    else:
                        print(f"  Facelet [{row}][{col}] failed to load")
                else:
                    print(f"  Facelet [{row}][{col}] not found: {facelet_file}")

        # Combine face and facelet grid side by side
        gap = 10
        gap_img = np.zeros((target_height, gap, 3), dtype=np.uint8)
        combined = np.hstack([face_img, gap_img, facelet_grid])

        # Add title
        cv2.putText(combined, side_name.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
