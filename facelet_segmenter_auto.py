"""
Rubik's Cube Face Segmentation Component - Auto-Select Version

Automatically determines the best segmentation algorithm based on image
characteristics. Analyzes the image to choose between:
- V5 (Brightness Otsu): Best for dark backgrounds with bright stickers
- V2 (Contour + Perspective): Best for complex backgrounds, tilted cubes

Selection criteria:
1. Background brightness: Dark backgrounds favor V5, bright backgrounds favor V2
2. Color saturation distribution: High saturation regions indicate stickers
3. Edge complexity: Many edges in background suggest V2 is better
4. Contrast between center and edges: High contrast favors V5

Facelet ordering:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from facelet_segmenter_v2 import FaceletSegmenterV2
from facelet_segmenter_v5 import FaceletSegmenterV5


@dataclass
class BoundingBox:
    """Represents a bounding box for the cube face region."""
    x: int
    y: int
    width: int
    height: int
    rotation: float = 0.0


@dataclass
class ImageAnalysis:
    """Results of analyzing an image to determine best segmenter."""
    background_brightness: float  # 0-255, average brightness of edges
    center_brightness: float      # 0-255, average brightness of center
    brightness_contrast: float    # Ratio of center to background brightness
    saturation_mean: float        # Average saturation
    saturation_std: float         # Saturation variation
    edge_density_background: float  # Edge pixels in background region
    edge_density_center: float    # Edge pixels in center region
    recommended_segmenter: str    # 'v2' or 'v5'
    confidence: float             # 0-1, how confident in the recommendation
    reason: str                   # Human-readable explanation


class FaceletSegmenterAuto:
    """
    Automatically selects the best segmentation algorithm based on image analysis.

    Usage:
        segmenter = FaceletSegmenterAuto(output_size=64)
        facelets = segmenter.segment(image)

        # To see which algorithm was selected:
        facelets, analysis = segmenter.segment_with_analysis(image)
        print(f"Used: {analysis.recommended_segmenter} - {analysis.reason}")
    """

    def __init__(self, output_size: int = 64, debug: bool = False):
        """
        Initialize the auto-selecting segmenter.

        Args:
            output_size: Size of output facelet images (default 64x64)
            debug: If True, print analysis details
        """
        self.output_size = output_size
        self.debug = debug

        # Initialize both segmenters
        self.segmenter_v2 = FaceletSegmenterV2(output_size=output_size)
        self.segmenter_v5 = FaceletSegmenterV5(output_size=output_size)

        # Track last analysis for debugging
        self.last_analysis: Optional[ImageAnalysis] = None

    def analyze_image(self, image: np.ndarray) -> ImageAnalysis:
        """
        Analyze image characteristics to determine best segmenter.

        Args:
            image: Input BGR image

        Returns:
            ImageAnalysis with recommendation
        """
        h, w = image.shape[:2]

        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define regions: center (where cube likely is) vs edges (background)
        margin_h = h // 4
        margin_w = w // 4

        # Background region (edges of image)
        bg_mask = np.ones((h, w), dtype=np.uint8)
        bg_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 0

        # Center region (where cube likely is)
        center_mask = np.zeros((h, w), dtype=np.uint8)
        center_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1

        # Brightness analysis (V channel)
        v_channel = hsv[:, :, 2]
        background_brightness = np.mean(v_channel[bg_mask == 1])
        center_brightness = np.mean(v_channel[center_mask == 1])

        # Avoid division by zero
        if background_brightness < 1:
            background_brightness = 1
        brightness_contrast = center_brightness / background_brightness

        # Saturation analysis
        s_channel = hsv[:, :, 1]
        saturation_mean = np.mean(s_channel)
        saturation_std = np.std(s_channel)

        # Edge detection for complexity analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density_background = np.sum(edges[bg_mask == 1]) / (255 * np.sum(bg_mask))
        edge_density_center = np.sum(edges[center_mask == 1]) / (255 * np.sum(center_mask))

        # Decision logic
        scores = {'v2': 0.0, 'v5': 0.0}
        reasons = []

        # Criterion 1: Background brightness
        # V5 works best with dark backgrounds (< 80), V2 handles bright backgrounds
        if background_brightness < 60:
            scores['v5'] += 0.4
            reasons.append("dark background favors V5")
        elif background_brightness < 100:
            scores['v5'] += 0.2
            scores['v2'] += 0.1
            reasons.append("medium background slightly favors V5")
        else:
            scores['v2'] += 0.4
            reasons.append("bright background favors V2")

        # Criterion 2: Brightness contrast between center and edges
        # V5 relies on brightness difference, needs high contrast
        if brightness_contrast > 1.3:
            scores['v5'] += 0.3
            reasons.append("high brightness contrast favors V5")
        elif brightness_contrast < 0.9:
            scores['v2'] += 0.2
            reasons.append("center darker than background favors V2")
        else:
            scores['v2'] += 0.1
            reasons.append("low contrast favors V2")

        # Criterion 3: Edge complexity in background
        # High edge density in background means cluttered scene -> V2 better
        if edge_density_background > 0.1:
            scores['v2'] += 0.3
            reasons.append("cluttered background favors V2")
        elif edge_density_background < 0.03:
            scores['v5'] += 0.2
            reasons.append("clean background favors V5")

        # Criterion 4: Saturation characteristics
        # High saturation with high std suggests colorful stickers against neutral bg
        if saturation_std > 50 and saturation_mean > 60:
            scores['v5'] += 0.1
            reasons.append("high saturation variation slightly favors V5")

        # Determine winner
        if scores['v5'] > scores['v2']:
            recommended = 'v5'
            confidence = min(1.0, (scores['v5'] - scores['v2']) / 0.5 + 0.5)
        else:
            recommended = 'v2'
            confidence = min(1.0, (scores['v2'] - scores['v5']) / 0.5 + 0.5)

        # Build reason string
        reason = f"Selected {recommended.upper()}: " + "; ".join(reasons[:2])

        return ImageAnalysis(
            background_brightness=background_brightness,
            center_brightness=center_brightness,
            brightness_contrast=brightness_contrast,
            saturation_mean=saturation_mean,
            saturation_std=saturation_std,
            edge_density_background=edge_density_background,
            edge_density_center=edge_density_center,
            recommended_segmenter=recommended,
            confidence=confidence,
            reason=reason
        )

    def segment(
        self,
        image: np.ndarray,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Segment a cube face image into 9 facelets using auto-selected algorithm.

        Args:
            image: Input image (BGR format from OpenCV)
            bbox: Optional bounding box (passed to underlying segmenter)

        Returns:
            numpy array of shape (3, 3, 64, 64, 3)
        """
        facelets, _ = self.segment_with_analysis(image, bbox)
        return facelets

    def segment_with_analysis(
        self,
        image: np.ndarray,
        bbox: Optional[BoundingBox] = None
    ) -> Tuple[np.ndarray, ImageAnalysis]:
        """
        Segment with full analysis results returned.

        Args:
            image: Input image (BGR format)
            bbox: Optional bounding box

        Returns:
            Tuple of (facelets array, ImageAnalysis)
        """
        # Analyze image
        analysis = self.analyze_image(image)
        self.last_analysis = analysis

        if self.debug:
            print(f"     [Auto] Background brightness: {analysis.background_brightness:.1f}")
            print(f"     [Auto] Center brightness: {analysis.center_brightness:.1f}")
            print(f"     [Auto] Brightness contrast: {analysis.brightness_contrast:.2f}")
            print(f"     [Auto] Edge density (bg): {analysis.edge_density_background:.3f}")
            print(f"     [Auto] {analysis.reason}")

        # Select and run appropriate segmenter
        if analysis.recommended_segmenter == 'v5':
            if self.debug:
                print("     [Auto] Using V5 (Brightness Otsu)")
            facelets = self.segmenter_v5.segment(image, bbox)
        else:
            if self.debug:
                print("     [Auto] Using V2 (Contour + Perspective)")
            facelets = self.segmenter_v2.segment(image, bbox)

        return facelets, analysis

    def segment_from_file(
        self,
        image_path: str,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Convenience method to segment directly from an image file.

        Args:
            image_path: Path to the input image
            bbox: Optional bounding box

        Returns:
            numpy array of shape (3, 3, 64, 64, 3)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        return self.segment(image, bbox)


def segment_cube_face(
    image: np.ndarray,
    bbox: Optional[BoundingBox] = None,
    output_size: int = 64
) -> np.ndarray:
    """
    Functional interface for auto-selecting facelet segmentation.

    Args:
        image: Input image containing a Rubik's cube face
        bbox: Optional bounding box
        output_size: Size of output facelet images (default 64)

    Returns:
        numpy array of shape (3, 3, output_size, output_size, 3)
    """
    segmenter = FaceletSegmenterAuto(output_size=output_size)
    return segmenter.segment(image, bbox)


# Quick test when run directly
if __name__ == "__main__":
    import os
    import sys

    # Test on all available test sets
    test_dirs = [
        "input_faces",
        "input_faces/Black Background",
        "input_faces/Grey_Background",
        "input_faces/Camera Captures"
    ]

    segmenter = FaceletSegmenterAuto(debug=True)

    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {test_dir}")
        print('='*60)

        for face in ['up', 'down', 'front', 'back', 'left', 'right']:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                path = os.path.join(test_dir, face + ext)
                if os.path.exists(path):
                    print(f"\n{face.upper()} face:")
                    image = cv2.imread(path)
                    if image is not None:
                        facelets, analysis = segmenter.segment_with_analysis(image)
                        print(f"  Result: {facelets.shape}")
                        print(f"  Confidence: {analysis.confidence:.2f}")
                    break
