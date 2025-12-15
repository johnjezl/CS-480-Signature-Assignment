"""
Rubik's Cube Face Segmentation Component - V5 (Greg's CV approach)

Uses HSV value channel with Otsu thresholding to detect bright stickers
against dark cube plastic. Based on Greg's Colab implementation.

Key differences from other segmenters:
- Uses V (brightness) channel instead of saturation or edges
- Otsu automatic thresholding for robust sticker detection
- K-means clustering to organize detected squares into 3 rows
- Works well when stickers are brighter than the cube body

Facelet ordering:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a bounding box for the cube face region."""
    x: int
    y: int
    width: int
    height: int
    rotation: float = 0.0


class FaceletSegmenterBrightnessOtsu:
    """
    Segments a Rubik's cube face using brightness-based detection.

    Uses Otsu thresholding on the V (value/brightness) channel of HSV
    to detect bright stickers against the dark cube plastic.

    Usage:
        segmenter = FaceletSegmenterBrightnessOtsu(output_size=64)
        facelets = segmenter.segment(image)
    """

    def __init__(self, output_size: int = 64, debug: bool = False):
        """
        Initialize the segmenter.

        Args:
            output_size: Size of output facelet images (default 64x64)
            debug: If True, print debug information during processing
        """
        self.output_size = output_size
        self.debug = debug
        self.work_width = 800  # Standard width for processing

    def segment(
        self,
        image: np.ndarray,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Segment a cube face image into 9 facelets.

        Args:
            image: Input image (BGR format from OpenCV)
                   Expected shape: (H, W, 3)
            bbox: Optional bounding box (not used in V5, kept for interface compatibility)

        Returns:
            numpy array of shape (3, 3, 64, 64, 3) representing a 3x3 grid of facelets
            Access pattern: [row][col][height][width][channel]
        """
        # Detect and extract the 9 facelet regions
        ordered_boxes, work_image, scale = self._detect_grid(image)

        if ordered_boxes is None or len(ordered_boxes) < 9:
            if self.debug:
                print(f"     [V5] Warning: Only detected {len(ordered_boxes) if ordered_boxes else 0} facelets, falling back to grid split")
            return self._fallback_grid_split(image)

        # Extract facelets from detected boxes
        facelets = self._extract_facelets_from_boxes(work_image, ordered_boxes)

        # Resize each facelet to output size
        resized_facelets = [
            cv2.resize(facelet, (self.output_size, self.output_size),
                      interpolation=cv2.INTER_AREA)
            for facelet in facelets
        ]

        # Reshape from list of 9 facelets to 3x3 grid
        facelets_array = np.array(resized_facelets, dtype=np.uint8)
        facelets_grid = facelets_array.reshape(3, 3, self.output_size, self.output_size, 3)

        return facelets_grid

    def _dedup_boxes(self, boxes: List[Tuple], min_dist: int = 20) -> List[Tuple]:
        """Remove near-duplicate boxes based on center distance."""
        kept = []
        for (x, y, w, h, area) in sorted(boxes, key=lambda b: b[4], reverse=True):
            cx = x + w / 2
            cy = y + h / 2
            if all(math.hypot(cx - (kx + kw/2), cy - (ky + kh/2)) > min_dist
                   for (kx, ky, kw, kh, ka) in kept):
                kept.append((x, y, w, h, area))
        return kept

    def _choose_nine(self, boxes: List[Tuple]) -> List[Tuple]:
        """From >9 boxes, keep the 9 closest to their centroid."""
        centers = [(x + w/2, y + h/2) for x, y, w, h, a in boxes]
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        dists = [math.hypot(c[0] - cx, c[1] - cy) for c in centers]
        idx_sorted = sorted(range(len(boxes)), key=lambda i: dists[i])
        return [boxes[i] for i in idx_sorted[:9]]

    def _detect_grid(self, image: np.ndarray) -> Tuple[Optional[List[Tuple]], np.ndarray, float]:
        """
        Detect the 9 facelets using brightness thresholding.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (ordered_boxes, work_image, scale)
            - ordered_boxes: List of 9 (x, y, w, h, area) tuples in reading order
            - work_image: Resized working image
            - scale: Scale factor used for resizing
        """
        h, w = image.shape[:2]
        scale = self.work_width / w
        work = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # Use brightness channel (stickers are bright vs dark plastic)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)

        # Otsu threshold on V channel
        _, thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate expected facelet size based on image dimensions
        # At 800px width, a cube face ~500px means facelets ~167px = ~28000 area
        # Allow range from ~5000 to ~50000 for various cube sizes/positions
        min_area = 3000
        max_area = 60000

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, bw, bh = cv2.boundingRect(c)
                # Roughly square (aspect ratio between 0.6 and 1.67)
                if 0.6 < bw / float(bh) < 1.67:
                    boxes.append((x, y, bw, bh, area))

        # Remove duplicates (inner/outer borders of same sticker)
        boxes = self._dedup_boxes(boxes, min_dist=20)

        if len(boxes) < 9:
            if self.debug:
                print(f"     [V5] Not enough squares: {len(boxes)}")
            return None, work, scale

        if len(boxes) > 9:
            boxes = self._choose_nine(boxes)

        # Sort into 3 rows using k-means on y centers
        centers = [(x + bw/2, y + bh/2) for x, y, bw, bh, a in boxes]
        ys = np.float32([c[1] for c in centers]).reshape(-1, 1)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        K = 3
        _, labels, centers_y = cv2.kmeans(
            ys, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        centers_y = centers_y.flatten()
        row_order = np.argsort(centers_y)  # top to bottom

        ordered_boxes = []
        for row_idx in row_order:
            row_boxes = [b for b, l in zip(boxes, labels) if l == row_idx]
            # Sort left to right within row
            row_boxes.sort(key=lambda b: b[0] + b[2]/2)
            ordered_boxes.extend(row_boxes)

        # Trim to 9 in case of any issues
        ordered_boxes = ordered_boxes[:9]

        if len(ordered_boxes) == 9 and self.debug:
            print(f"     [V5] Detected 9 facelets successfully")

        return ordered_boxes, work, scale

    def _extract_facelets_from_boxes(
        self,
        work_image: np.ndarray,
        boxes: List[Tuple]
    ) -> List[np.ndarray]:
        """
        Extract facelet images from detected bounding boxes.

        Args:
            work_image: The working (resized) image
            boxes: List of (x, y, w, h, area) tuples

        Returns:
            List of 9 facelet images
        """
        facelets = []
        for (x, y, w, h, area) in boxes:
            # Add small margin inward to avoid borders
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)

            x1 = x + margin_x
            y1 = y + margin_y
            x2 = x + w - margin_x
            y2 = y + h - margin_y

            # Ensure valid bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(work_image.shape[1], x2)
            y2 = min(work_image.shape[0], y2)

            facelet = work_image[y1:y2, x1:x2].copy()
            facelets.append(facelet)

        return facelets

    def _fallback_grid_split(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback method: split image into 3x3 grid when detection fails.

        Args:
            image: Input image

        Returns:
            numpy array of shape (3, 3, 64, 64, 3)
        """
        height, width = image.shape[:2]

        # Use center region (assume cube is centered)
        min_dim = min(height, width)
        margin = int(min_dim * 0.1)

        # Calculate cube region (centered square)
        size = min_dim - 2 * margin
        start_x = (width - size) // 2
        start_y = (height - size) // 2

        face_region = image[start_y:start_y+size, start_x:start_x+size]

        # Split into 3x3 grid
        facelet_h = size // 3
        facelet_w = size // 3

        facelets = []
        for row in range(3):
            for col in range(3):
                y1 = row * facelet_h
                y2 = (row + 1) * facelet_h if row < 2 else size
                x1 = col * facelet_w
                x2 = (col + 1) * facelet_w if col < 2 else size

                # Add margin to avoid borders
                margin_h = int(facelet_h * 0.08)
                margin_w = int(facelet_w * 0.08)

                y1 = min(y1 + margin_h, y2 - 1)
                y2 = max(y2 - margin_h, y1 + 1)
                x1 = min(x1 + margin_w, x2 - 1)
                x2 = max(x2 - margin_w, x1 + 1)

                facelet = face_region[y1:y2, x1:x2].copy()
                facelet = cv2.resize(facelet, (self.output_size, self.output_size),
                                    interpolation=cv2.INTER_AREA)
                facelets.append(facelet)

        facelets_array = np.array(facelets, dtype=np.uint8)
        return facelets_array.reshape(3, 3, self.output_size, self.output_size, 3)

    def segment_from_file(
        self,
        image_path: str,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Convenience method to segment directly from an image file.

        Args:
            image_path: Path to the input image
            bbox: Optional bounding box (not used in V5)

        Returns:
            numpy array of shape (3, 3, 64, 64, 3)
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
    Functional interface for facelet segmentation using V5 method.

    Args:
        image: Input image containing a Rubik's cube face
               Expected shape: (H, W, 3) in BGR format
        bbox: Optional bounding box (not used in V5)
        output_size: Size of output facelet images (default 64)

    Returns:
        numpy array of shape (3, 3, output_size, output_size, 3)
    """
    segmenter = FaceletSegmenterBrightnessOtsu(output_size=output_size)
    return segmenter.segment(image, bbox)
