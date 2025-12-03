"""
Interactive Facelet Extractor Tool

Extract 64x64 facelet patches from Rubik's cube images for training data.

Usage:
    python interactive_facelet_extrator_tool.py <image_files...>
    python interactive_facelet_extrator_tool.py *.jpg
    python interactive_facelet_extrator_tool.py image1.jpg image2.png ../photos/*.jpg

Controls:
    - Click center of facelet to extract 64x64 region
    - Press 'r/o/y/g/b/w' to set color class (red/orange/yellow/green/blue/white)
    - Press 'n' or 'q' to move to next image
    - Press 'p' to go to previous image
    - Press ESC to exit immediately
"""

import cv2
import numpy as np
import os
import sys
import glob
from datetime import datetime


class FaceletExtractor:
    def __init__(self, output_dir="facelet_dataset", max_display_size=1000):
        self.output_dir = output_dir
        self.current_image = None
        self.original_image = None  # Keep original for reset
        self.display_image = None   # Scaled version for display
        self.scale_factor = 1.0     # Scale factor from display to original
        self.max_display_size = max_display_size
        self.window_name = "Facelet Extractor - Click to Extract 64x64"

        # Create output directories for each color
        self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'white']
        for color in self.colors:
            os.makedirs(f"{output_dir}/{color}", exist_ok=True)

        self.current_color = 'red'
        self.extracted_count = 0

    def _scale_image_for_display(self, image):
        """Scale image to fit within max_display_size while preserving aspect ratio."""
        h, w = image.shape[:2]
        if max(h, w) <= self.max_display_size:
            self.scale_factor = 1.0
            return image.copy()

        # Calculate scale factor
        self.scale_factor = self.max_display_size / max(h, w)
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def extract_facelets_from_list(self, image_paths):
        """Process a list of images with navigation between them."""
        if not image_paths:
            print("No images to process.")
            return

        current_index = 0

        while 0 <= current_index < len(image_paths):
            image_path = image_paths[current_index]
            print(f"\n[{current_index + 1}/{len(image_paths)}] Loading: {image_path}")

            result = self.extract_facelets(image_path, current_index + 1, len(image_paths))

            if result == "next":
                current_index += 1
            elif result == "prev":
                current_index = max(0, current_index - 1)
            elif result == "exit":
                break

        print(f"\nDone! Extracted {self.extracted_count} facelets total.")
        cv2.destroyAllWindows()

    def extract_facelets(self, image_path, current_num=1, total_num=1):
        """Extract 64x64 patches from image with mouse clicks.

        Returns:
            "next" - move to next image
            "prev" - move to previous image
            "exit" - exit the tool
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Could not load {image_path}")
            return "next"

        self.current_image = self.original_image.copy()
        self.display_image = self._scale_image_for_display(self.current_image)

        orig_h, orig_w = self.original_image.shape[:2]
        scale_info = f" (scaled to {self.scale_factor:.0%})" if self.scale_factor < 1.0 else ""

        print("\nControls:")
        print("  Click  - Extract 64x64 facelet at click location")
        print("  r/o/y/g/b/w - Set color class")
        print("  n or q - Next image")
        print("  p      - Previous image")
        print("  z      - Reset (clear rectangles)")
        print("  ESC    - Exit tool")
        print(f"\nImage size: {orig_w}x{orig_h}{scale_info}")
        print(f"Current color: {self.current_color}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            # Update display image from current_image (which has rectangles drawn)
            self.display_image = self._scale_image_for_display(self.current_image)
            display = self.display_image.copy()

            # Draw status bar at top
            status_text = f"[{current_num}/{total_num}] Color: {self.current_color.upper()} | Extracted: {self.extracted_count}"
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(display, status_text, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show filename and scale at bottom
            filename = os.path.basename(image_path)
            bottom_text = f"{filename} ({orig_w}x{orig_h})"
            if self.scale_factor < 1.0:
                bottom_text += f" [{self.scale_factor:.0%}]"
            cv2.rectangle(display, (0, display.shape[0] - 30), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
            cv2.putText(display, bottom_text, (10, display.shape[0] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            # Color selection
            if key == ord('r'):
                self.current_color = 'red'
                print(f"Color: {self.current_color}")
            elif key == ord('o'):
                self.current_color = 'orange'
                print(f"Color: {self.current_color}")
            elif key == ord('y'):
                self.current_color = 'yellow'
                print(f"Color: {self.current_color}")
            elif key == ord('g'):
                self.current_color = 'green'
                print(f"Color: {self.current_color}")
            elif key == ord('b'):
                self.current_color = 'blue'
                print(f"Color: {self.current_color}")
            elif key == ord('w'):
                self.current_color = 'white'
                print(f"Color: {self.current_color}")

            # Navigation
            elif key == ord('n') or key == ord('q'):
                return "next"
            elif key == ord('p'):
                return "prev"
            elif key == 27:  # ESC
                return "exit"

            # Reset image (clear drawn rectangles)
            elif key == ord('z'):
                self.current_image = self.original_image.copy()
                print("Reset image")

    def mouse_callback(self, event, x, y, flags, param):
        """Extract 64x64 region centered at click."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map click coordinates from display to original image
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)

            h, w = self.current_image.shape[:2]

            # Calculate bounds (ensure we don't go out of image)
            x1 = max(0, orig_x - 32)
            y1 = max(0, orig_y - 32)
            x2 = min(w, x1 + 64)
            y2 = min(h, y1 + 64)

            # Adjust if near edge
            if x2 - x1 < 64:
                x1 = max(0, x2 - 64)
            if y2 - y1 < 64:
                y1 = max(0, y2 - 64)

            # Ensure we have a valid 64x64 region
            if x2 - x1 < 64 or y2 - y1 < 64:
                print("Too close to edge - cannot extract 64x64 region")
                return

            # Extract facelet from original resolution image
            facelet = self.original_image[y1:y2, x1:x2]

            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.output_dir}/{self.current_color}/facelet_{timestamp}.png"
            cv2.imwrite(filename, facelet)

            self.extracted_count += 1

            # Show preview
            preview = cv2.resize(facelet, (128, 128), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Last Extracted", preview)
            print(f"  Saved: {filename}")

            # Draw rectangle on current_image (full resolution) to show what was extracted
            # Color code the rectangle based on current color
            color_bgr = {
                'red': (0, 0, 255),
                'orange': (0, 165, 255),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'white': (255, 255, 255)
            }
            rect_color = color_bgr.get(self.current_color, (0, 255, 0))
            # Scale line thickness based on image size
            thickness = max(2, int(3 / self.scale_factor))
            cv2.rectangle(self.current_image, (x1, y1), (x2, y2), rect_color, thickness)


def expand_wildcards(patterns):
    """Expand wildcard patterns to actual file paths."""
    files = []
    for pattern in patterns:
        # Use glob to expand wildcards
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.exists(pattern):
            # If no wildcard and file exists, add it directly
            files.append(pattern)
        else:
            print(f"Warning: No files match '{pattern}'")

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: No image files specified.")
        print("\nExamples:")
        print("  python interactive_facelet_extrator_tool.py image.jpg")
        print("  python interactive_facelet_extrator_tool.py *.jpg *.png")
        print("  python interactive_facelet_extrator_tool.py ../input_faces/*.jpg")
        sys.exit(1)

    # Expand wildcards in command line arguments
    image_paths = expand_wildcards(sys.argv[1:])

    if not image_paths:
        print("Error: No valid image files found.")
        sys.exit(1)

    # Filter to only image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_extensions]

    if not image_paths:
        print("Error: No valid image files found (supported: jpg, jpeg, png, bmp, tiff)")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process:")
    for i, path in enumerate(image_paths[:5]):
        print(f"  {i+1}. {path}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")

    # Determine output directory (next to the script or in current directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "datasets", "real_facelets")
    output_dir = os.path.normpath(output_dir)

    print(f"\nOutput directory: {output_dir}")

    extractor = FaceletExtractor(output_dir)
    extractor.extract_facelets_from_list(image_paths)


if __name__ == "__main__":
    main()
