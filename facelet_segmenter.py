"""
Rubik's Cube Face Segmentation Component

Takes an image containing a Rubik's cube face and outputs 9 segmented
64x64 images of each facelet, ordered top-left to bottom-right.

Facelet ordering:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a bounding box for the cube face region."""
    x: int
    y: int
    width: int
    height: int


class FaceletSegmenter:
    """
    Segments a Rubik's cube face image into 9 individual facelet images.

    Usage:
        segmenter = FaceletSegmenter(output_size=64)
        facelets = segmenter.segment(image)
        # or with explicit bounding box:
        facelets = segmenter.segment(image, bbox=BoundingBox(100, 50, 300, 300))
    """

    def __init__(self, output_size: int = 64):
        """
        Initialize the segmenter.

        Args:
            output_size: Size of output facelet images (default 64x64)
        """
        self.output_size = output_size

    def segment(
        self,
        image: np.ndarray,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Segment a cube face image into 9 facelets.

        Args:
            image: Input image (BGR format from OpenCV or RGB)
                   Expected shape: (H, W, 3) where H=height, W=width, 3=BGR channels
            bbox: Optional bounding box for the cube face region.
                  If None, attempts to auto-detect or uses full image.

        Returns:
            numpy array of shape (3, 3, 64, 64, 3) representing a 3x3 grid of facelets
            Access pattern: [row][col][height][width][channel]
            Ordering: row 0 = top row, col 0 = left column
        """
        if bbox is None:
            bbox = self._detect_face_region(image)

        # Extract the face region
        face_region = self._extract_region(image, bbox)

        # Split into 9 facelets
        facelets = self._split_into_facelets(face_region)

        # Resize each facelet to output size
        resized_facelets = [
            cv2.resize(facelet, (self.output_size, self.output_size),
                      interpolation=cv2.INTER_AREA)
            for facelet in facelets
        ]

        # Reshape from list of 9 facelets to 3x3 grid
        # Convert list to array: (9, 64, 64, 3)
        facelets_array = np.array(resized_facelets, dtype=np.uint8)

        # Reshape to 3x3 grid: (3, 3, 64, 64, 3)
        facelets_grid = facelets_array.reshape(3, 3, self.output_size, self.output_size, 3)

        return facelets_grid

    def _detect_face_region(self, image: np.ndarray) -> BoundingBox:
        """
        Attempt to auto-detect the cube face region in the image.

        Uses color saturation and contour analysis to find the colorful cube region.
        Falls back to using a centered square if detection fails.

        Args:
            image: Input image

        Returns:
            BoundingBox for the detected face region
        """
        height, width = image.shape[:2]

        # Convert to HSV for color-based detection
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            # Grayscale image, use fallback
            size = min(width, height) // 2
            x = (width - size) // 2
            y = (height - size) // 2
            return BoundingBox(x, y, size, size)

        # Use saturation channel to find colorful regions (Rubik's cubes are colorful)
        saturation = hsv[:, :, 1]

        # Threshold saturation to find colorful areas
        _, sat_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in saturation mask
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = 0

        # Expected cube size range (as fraction of image dimensions)
        min_size = min(width, height) * 0.1  # At least 10% of smallest dimension
        max_size = min(width, height) * 0.8  # At most 80% of smallest dimension

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size * min_size:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if w < min_size or h < min_size or w > max_size or h > max_size:
                continue

            # Calculate squareness (cube faces should be square-ish)
            squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0

            if squareness < 0.6:  # Too rectangular, not cube-like
                continue

            # Score based on size, squareness, and color saturation
            roi_sat = saturation[y:y+h, x:x+w]
            avg_saturation = roi_sat.mean()

            # Prefer medium-sized, square regions with high color saturation
            size_score = (w * h) / (max_size * max_size)  # Normalize by max size
            score = squareness * size_score * (avg_saturation / 255.0)

            if score > best_score:
                best_score = score
                best_bbox = BoundingBox(x, y, w, h)

        # Fallback: use centered square region
        if best_bbox is None:
            # Use a reasonable centered square (about 1/3 of image size)
            size = min(width, height) // 3
            x = (width - size) // 2
            y = (height - size) // 2
            best_bbox = BoundingBox(x, y, size, size)

        return best_bbox

    def _extract_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract and square-crop the face region from the image.

        Args:
            image: Input image
            bbox: Bounding box for extraction

        Returns:
            Extracted region as numpy array
        """
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Ensure bounds are within image
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        # Extract the region
        region = image[y:y+h, x:x+w]

        # Make it square by cropping to the smaller dimension
        min_dim = min(region.shape[0], region.shape[1])

        # Center crop to square
        start_y = (region.shape[0] - min_dim) // 2
        start_x = (region.shape[1] - min_dim) // 2

        square_region = region[start_y:start_y+min_dim, start_x:start_x+min_dim]

        return square_region

    def _split_into_facelets(self, face_region: np.ndarray) -> List[np.ndarray]:
        """
        Split the square face region into 9 facelets, avoiding black borders.

        Args:
            face_region: Square image of the cube face

        Returns:
            List of 9 facelet images, ordered top-left to bottom-right
        """
        height, width = face_region.shape[:2]

        # Calculate base facelet dimensions
        facelet_h = height // 3
        facelet_w = width // 3

        # Add margin to avoid black borders between facelets
        # Typically, borders are about 5-10% of facelet size
        margin_h = int(facelet_h * 0.08)  # 8% margin
        margin_w = int(facelet_w * 0.08)

        facelets = []

        # Extract each facelet (row by row, left to right)
        for row in range(3):
            for col in range(3):
                # Base coordinates
                y_start = row * facelet_h
                y_end = (row + 1) * facelet_h if row < 2 else height
                x_start = col * facelet_w
                x_end = (col + 1) * facelet_w if col < 2 else width

                # Apply margin to avoid borders
                # For edges touching the outer boundary, use less margin
                top_margin = margin_h if row > 0 else margin_h // 2
                bottom_margin = margin_h if row < 2 else margin_h // 2
                left_margin = margin_w if col > 0 else margin_w // 2
                right_margin = margin_w if col < 2 else margin_w // 2

                # Apply margins
                y_start = min(y_start + top_margin, y_end - 1)
                y_end = max(y_end - bottom_margin, y_start + 1)
                x_start = min(x_start + left_margin, x_end - 1)
                x_end = max(x_end - right_margin, x_start + 1)

                facelet = face_region[y_start:y_end, x_start:x_end].copy()
                facelets.append(facelet)

        return facelets

    def segment_from_file(
        self,
        image_path: str,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Convenience method to segment directly from an image file.

        Args:
            image_path: Path to the input image
            bbox: Optional bounding box for the cube face region

        Returns:
            numpy array of shape (3, 3, 64, 64, 3) representing a 3x3 grid of facelets
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        return self.segment(image, bbox)

    def save_facelets(
        self,
        facelets: np.ndarray,
        output_dir: str,
        prefix: str = "facelet"
    ) -> List[str]:
        """
        Save facelet images to files.

        Args:
            facelets: Grid of facelets with shape (3, 3, 64, 64, 3)
            output_dir: Directory to save images
            prefix: Filename prefix (default "facelet")

        Returns:
            List of saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        idx = 0
        for row in range(3):
            for col in range(3):
                facelet = facelets[row, col]
                path = os.path.join(output_dir, f"{prefix}_{idx}.png")
                cv2.imwrite(path, facelet)
                saved_paths.append(path)
                idx += 1

        return saved_paths


def segment_cube_face(
    image: np.ndarray,
    bbox: Optional[BoundingBox] = None,
    output_size: int = 64
) -> np.ndarray:
    """
    Functional interface for facelet segmentation.

    Args:
        image: Input image containing a Rubik's cube face
               Expected shape: (H, W, 3) in BGR format
        bbox: Optional bounding box for the face region
        output_size: Size of output facelet images (default 64)

    Returns:
        numpy array of shape (3, 3, output_size, output_size, 3)
        representing a 3x3 grid of facelets in BGR format
    """
    segmenter = FaceletSegmenter(output_size=output_size)
    return segmenter.segment(image, bbox)
