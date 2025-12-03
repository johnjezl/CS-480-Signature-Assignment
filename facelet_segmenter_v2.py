"""
Rubik's Cube Face Segmentation Component - Version 2

Improved segmentation using:
1. Contour-based cube face detection with quadrilateral fitting
2. Perspective correction via homography transform
3. LAB color space for better color boundary detection
4. Robust grid line detection with Hough transforms and RANSAC-style filtering
5. Multi-strategy detection with fallbacks

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
    rotation: float = 0.0  # Rotation angle in degrees (counter-clockwise)


@dataclass
class Quadrilateral:
    """Represents a 4-point polygon for perspective correction."""
    points: np.ndarray  # Shape (4, 2) - corners in order: TL, TR, BR, BL

    def to_bbox(self) -> BoundingBox:
        """Convert to axis-aligned bounding box."""
        x_min = int(np.min(self.points[:, 0]))
        y_min = int(np.min(self.points[:, 1]))
        x_max = int(np.max(self.points[:, 0]))
        y_max = int(np.max(self.points[:, 1]))
        return BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)


class FaceletSegmenterV2:
    """
    Version 2 of the Rubik's cube face segmenter with improved detection.

    Key improvements over v1:
    - Contour-based quadrilateral detection for cube face boundary
    - Perspective correction to handle tilted cubes
    - LAB color space for better color boundary detection
    - Robust grid line detection with outlier rejection
    - Multi-strategy approach with fallbacks

    Usage:
        segmenter = FaceletSegmenterV2(output_size=64)
        facelets = segmenter.segment(image)
        # or with explicit bounding box (falls back to v1-style):
        facelets = segmenter.segment(image, bbox=BoundingBox(100, 50, 300, 300))
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
            bbox: Optional bounding box for the cube face region.
                  If None, uses advanced detection to find the cube.

        Returns:
            numpy array of shape (3, 3, 64, 64, 3) representing a 3x3 grid
            Access pattern: [row][col][height][width][channel]
        """
        if bbox is not None:
            # Use provided bounding box - extract and process
            face_region = self._extract_bbox_region(image, bbox)
        else:
            # Use advanced detection pipeline
            face_region = self._detect_and_extract_face(image)

        # Split into 9 facelets with smart boundary detection
        facelets = self._extract_facelets_smart(face_region)

        # Resize each facelet to output size
        resized_facelets = [
            cv2.resize(facelet, (self.output_size, self.output_size),
                      interpolation=cv2.INTER_AREA)
            for facelet in facelets
        ]

        # Reshape to 3x3 grid
        facelets_array = np.array(resized_facelets, dtype=np.uint8)
        facelets_grid = facelets_array.reshape(3, 3, self.output_size, self.output_size, 3)

        return facelets_grid

    def _detect_and_extract_face(self, image: np.ndarray) -> np.ndarray:
        """
        Main detection pipeline - tries multiple strategies.

        Key insight: We must verify that the detected region contains a 3x3 grid
        of facelets separated by black borders, not just a single colored region.

        Strategy order:
        1. Grid-validated detection (find region AND verify 3x3 grid structure)
        2. Contour-based with grid validation
        3. Color saturation with grid validation
        4. Fallback to center crop with grid validation
        """
        height, width = image.shape[:2]

        if self.debug:
            print(f"[V2] Processing image: {width}x{height}")

        # Calculate minimum cube size - cube should be at least 20% of image
        # This allows for cubes that are smaller in frame while still
        # rejecting individual facelets (which are ~7% of image)
        min_cube_size = int(min(width, height) * 0.20)

        # Strategy 1: Try contour-based quadrilateral detection with grid validation
        quad = self._detect_quadrilateral(image, min_size=min_cube_size)
        if quad is not None:
            region = self._extract_perspective_corrected(image, quad)
            if self._validate_has_grid(region):
                if self.debug:
                    print("[V2] Using quadrilateral detection (grid validated)")
                return region
            elif self.debug:
                print("[V2] Quadrilateral detection failed grid validation")

        # Strategy 2: Color saturation + contour detection with grid validation
        color_quad = self._detect_by_color_saturation(image, min_size=min_cube_size)
        if color_quad is not None:
            region = self._extract_perspective_corrected(image, color_quad)
            if self._validate_has_grid(region):
                if self.debug:
                    print("[V2] Using color saturation detection (grid validated)")
                return region
            elif self.debug:
                print("[V2] Color saturation detection failed grid validation")

        # Strategy 3: Try grid-based detection (inherently validates grid)
        grid_region = self._detect_by_grid_lines(image)
        if grid_region is not None:
            if self.debug:
                print("[V2] Using grid line detection")
            return grid_region

        # Strategy 4: Fallback - use center crop with grid validation
        if self.debug:
            print("[V2] Trying fallback center crop")
        fallback_region = self._fallback_center_crop(image)
        if self._validate_has_grid(fallback_region):
            if self.debug:
                print("[V2] Using fallback center crop (grid validated)")
            return fallback_region

        # Ultimate fallback - return the best attempt even without validation
        if self.debug:
            print("[V2] WARNING: No grid validation passed, using best effort")
        return fallback_region

    def _validate_has_grid(self, region: np.ndarray) -> bool:
        """
        Validate that a region contains a 3x3 grid structure.

        Looks for 2 horizontal and 2 vertical dark lines (black plastic borders)
        that divide the region into 9 roughly equal parts.
        Also verifies the 9 cells have color variance (not all same color).

        Returns:
            True if a valid 3x3 grid is detected, False otherwise.
        """
        if region is None or region.size == 0:
            return False

        height, width = region.shape[:2]
        if height < 30 or width < 30:
            return False

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Find dark pixels (black borders between facelets)
        # The black plastic borders are typically < 50 brightness
        # Use a fixed threshold that works for black plastic
        # Also try adaptive if fixed doesn't work well
        dark_threshold = 50  # Black plastic is usually < 50
        _, dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

        # Project onto horizontal and vertical axes
        h_projection = np.sum(dark_mask, axis=1).astype(float)
        v_projection = np.sum(dark_mask, axis=0).astype(float)

        # Smooth projections
        kernel_size = max(3, height // 30)
        kernel = np.ones(kernel_size) / kernel_size
        h_projection = np.convolve(h_projection, kernel, mode='same')
        v_projection = np.convolve(v_projection, kernel, mode='same')

        # Find peaks (dark line positions)
        h_peaks = self._find_grid_peaks(h_projection, height)
        v_peaks = self._find_grid_peaks(v_projection, width)

        # We need at least 2 internal lines in each direction (creating 3 sections)
        if len(h_peaks) < 2 or len(v_peaks) < 2:
            if self.debug:
                print(f"[V2] Grid validation: insufficient peaks (h={len(h_peaks)}, v={len(v_peaks)})")
            return False

        # Check that the peaks divide the region into roughly equal thirds
        h_valid = self._check_even_spacing(h_peaks, height)
        v_valid = self._check_even_spacing(v_peaks, width)

        if not (h_valid and v_valid):
            if self.debug:
                print(f"[V2] Grid validation: spacing invalid (h_valid={h_valid}, v_valid={v_valid})")
            return False

        # Additional check: verify the 9 cells have color variance
        # A real cube face should have at least 2 different colors (usually more)
        color_variance = self._check_color_variance(region)

        if self.debug:
            print(f"[V2] Grid validation: h_peaks={len(h_peaks)}, v_peaks={len(v_peaks)}, color_var={color_variance:.1f}")

        # Require minimum color variance - single facelet will have very low variance
        # A typical scrambled cube face has variance > 1000
        if color_variance < 500:
            if self.debug:
                print(f"[V2] Grid validation: color variance too low ({color_variance:.0f} < 500)")
            return False

        return True

    def _check_color_variance(self, region: np.ndarray) -> float:
        """
        Check the color variance across a region.

        Splits the region into 9 cells and measures how different
        the average colors are between cells.

        Returns:
            Variance score (higher = more color diversity)
        """
        height, width = region.shape[:2]
        cell_h = height // 3
        cell_w = width // 3

        # Get average color of each cell
        cell_colors = []
        for row in range(3):
            for col in range(3):
                y1 = row * cell_h + cell_h // 4
                y2 = (row + 1) * cell_h - cell_h // 4
                x1 = col * cell_w + cell_w // 4
                x2 = (col + 1) * cell_w - cell_w // 4

                cell = region[y1:y2, x1:x2]
                avg_color = cell.mean(axis=(0, 1))
                cell_colors.append(avg_color)

        cell_colors = np.array(cell_colors)

        # Calculate variance across cells
        variance = np.var(cell_colors, axis=0).sum()

        return variance

    def _find_grid_peaks(self, projection: np.ndarray, dimension: int) -> List[int]:
        """
        Find peaks in a projection that correspond to grid lines.

        Returns positions of internal grid lines (not edges).
        """
        # Normalize projection
        if projection.max() == 0:
            return []

        projection = projection / projection.max()

        # Find local maxima above threshold
        threshold = 0.3
        margin = dimension // 6  # Don't look for lines too close to edges

        peaks = []
        window = max(3, dimension // 20)

        for i in range(margin, dimension - margin):
            if projection[i] > threshold:
                # Check if it's a local maximum
                start = max(0, i - window)
                end = min(dimension, i + window + 1)
                if projection[i] >= np.max(projection[start:end]) - 0.05:
                    # Check we're not too close to an existing peak
                    if not peaks or i - peaks[-1] > dimension // 6:
                        peaks.append(i)

        return peaks

    def _check_even_spacing(self, peaks: List[int], dimension: int) -> bool:
        """
        Check if peaks divide the dimension into roughly equal thirds.

        Handles cases where the extracted region has extra margin (e.g., black
        space around the cube) by looking for any two peaks that are roughly
        1/3 of the dimension apart.
        """
        if len(peaks) < 2:
            return False

        peaks = sorted(peaks)

        # Strategy 1: Check if peaks match expected 1/3 and 2/3 positions
        target1 = dimension / 3
        target2 = 2 * dimension / 3
        tolerance = dimension * 0.20

        best1 = min(peaks, key=lambda p: abs(p - target1))
        best2 = min(peaks, key=lambda p: abs(p - target2))

        if best1 != best2:
            spacing = best2 - best1
            expected_spacing = dimension / 3
            if (abs(best1 - target1) <= tolerance and
                abs(best2 - target2) <= tolerance and
                abs(spacing - expected_spacing) <= tolerance):
                return True

        # Strategy 2: If we have 2+ peaks, check if ANY consecutive pair
        # has spacing close to 1/3 of the dimension (grid line spacing)
        # This handles cases where the region has extra margin
        expected_spacing = dimension / 3
        spacing_tolerance = dimension * 0.15  # Tighter tolerance for spacing

        for i in range(len(peaks) - 1):
            spacing = peaks[i + 1] - peaks[i]
            if abs(spacing - expected_spacing) <= spacing_tolerance:
                # Found one valid spacing, check if there's another
                for j in range(i + 1, len(peaks) - 1):
                    spacing2 = peaks[j + 1] - peaks[j]
                    if abs(spacing2 - expected_spacing) <= spacing_tolerance:
                        # Two consecutive spacings match expected grid spacing
                        return True

        # Strategy 3: Check if we have exactly 2 peaks with correct spacing
        if len(peaks) == 2:
            spacing = peaks[1] - peaks[0]
            if abs(spacing - expected_spacing) <= tolerance:
                return True

        return False

    def _detect_quadrilateral(self, image: np.ndarray, min_size: int = 0) -> Optional[Quadrilateral]:
        """
        Detect the cube face as a quadrilateral using contour detection.

        Uses edge detection and contour approximation to find the
        largest approximately square 4-sided polygon.

        Args:
            image: Input image
            min_size: Minimum side length for detected cube (pixels)
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multi-scale edge detection for robustness
        edges_list = []
        for sigma in [1.0, 1.5, 2.0]:
            ksize = int(6 * sigma + 1) | 1  # Ensure odd
            blurred = cv2.GaussianBlur(filtered, (ksize, ksize), sigma)
            edges = cv2.Canny(blurred, 30, 100)
            edges_list.append(edges)

        # Combine edges from all scales
        combined_edges = np.zeros_like(edges_list[0])
        for edges in edges_list:
            combined_edges = cv2.bitwise_or(combined_edges, edges)

        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(combined_edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter and score contours - use min_size if provided
        min_dim = max(min_size, int(min(width, height) * 0.15))
        min_area = min_dim ** 2
        max_area = (min(width, height) * 0.85) ** 2

        best_quad = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # We want a quadrilateral
            if len(approx) != 4:
                # Try with different epsilon values
                for eps in [0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, eps * peri, True)
                    if len(approx) == 4:
                        break

            if len(approx) != 4:
                continue

            # Order points: TL, TR, BR, BL
            points = self._order_quadrilateral_points(approx.reshape(4, 2))

            # Check if it's roughly square
            squareness = self._calculate_squareness(points)
            if squareness < 0.7:
                continue

            # Score based on area, squareness, and position
            # Prefer larger, more square regions near center
            cx, cy = np.mean(points, axis=0)
            center_dist = np.sqrt((cx - width/2)**2 + (cy - height/2)**2)
            center_score = 1.0 - (center_dist / (np.sqrt(width**2 + height**2) / 2))

            score = (area / max_area) * (squareness ** 2) * (0.5 + 0.5 * center_score)

            if score > best_score:
                best_score = score
                best_quad = Quadrilateral(points)

        return best_quad

    def _detect_by_color_saturation(self, image: np.ndarray, min_size: int = 0) -> Optional[Quadrilateral]:
        """
        Detect cube face by finding highly saturated (colorful) regions.

        Rubik's cubes are typically the most colorful objects in frame.
        This works well with dark or neutral backgrounds.

        Note: White/gray facelets have low saturation, so after finding the
        colorful region, we expand it to include adjacent areas bounded by
        the black plastic frame.

        Args:
            image: Input image
            min_size: Minimum side length for detected cube (pixels)
        """
        height, width = image.shape[:2]

        # Convert to HSV and LAB for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Saturation channel
        saturation = hsv[:, :, 1]

        # Create mask of colorful regions
        # Use adaptive threshold based on image statistics
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        sat_threshold = max(30, sat_mean + 0.5 * sat_std)

        _, sat_mask = cv2.threshold(saturation, sat_threshold, 255, cv2.THRESH_BINARY)

        # Also detect black plastic borders (low saturation, low lightness)
        # which separate facelets
        lightness = lab[:, :, 0]
        _, dark_mask = cv2.threshold(lightness, 40, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up the saturation mask
        kernel = np.ones((5, 5), np.uint8)
        sat_mask_clean = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        sat_mask_clean = cv2.morphologyEx(sat_mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

        # Try to fill holes inside the colored region, but be careful with dark backgrounds
        # First check if corners are dark (black background)
        # Sample multiple points near each corner for robustness
        margin = min(20, height // 50, width // 50)
        corner_samples = []
        for cy, cx in [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)]:
            y1, y2 = max(0, cy-margin), min(height, cy+margin+1)
            x1, x2 = max(0, cx-margin), min(width, cx+margin+1)
            corner_samples.append(np.mean(lightness[y1:y2, x1:x2]))

        is_dark_background = np.mean(corner_samples) < 80

        if is_dark_background:
            # For dark backgrounds, combine saturation (colored facelets) with
            # brightness (white facelets) to get complete cube coverage
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # White facelets are bright (>180) on dark background
            _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            # Combine colored and white regions
            color_and_white = cv2.bitwise_or(sat_mask_clean, bright_mask)
            # Morphological closing to connect nearby regions
            large_kernel = np.ones((15, 15), np.uint8)
            combined_mask = cv2.morphologyEx(color_and_white, cv2.MORPH_CLOSE, large_kernel, iterations=2)
        else:
            # For lighter or mixed backgrounds, use flood fill to find enclosed regions
            filled = sat_mask_clean.copy()
            h, w = filled.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(filled, flood_mask, (0, 0), 255)
            cv2.floodFill(filled, flood_mask, (w-1, 0), 255)
            cv2.floodFill(filled, flood_mask, (0, h-1), 255)
            cv2.floodFill(filled, flood_mask, (w-1, h-1), 255)
            filled_inv = cv2.bitwise_not(filled)
            combined_mask = cv2.bitwise_or(sat_mask_clean, filled_inv)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour that could be the cube - use min_size if provided
        min_dim = max(min_size, int(min(width, height) * 0.2))
        min_area = min_dim ** 2
        max_area = (min(width, height) * 0.9) ** 2

        best_quad = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Get the bounding rectangle (axis-aligned)
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the contour has irregular shape (potential partial view of other faces)
            # Compare contour area to bounding rect area - if significantly less,
            # there may be "tails" extending from the main cube face
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0

            # For a clean cube face, fill ratio should be high (>0.85)
            # For a cube with partial other faces visible, it will be lower
            has_irregular_shape = fill_ratio < 0.80

            if has_irregular_shape:
                # Use the largest inscribed square within the contour
                # This ignores "tails" from partial views of adjacent faces
                points = self._find_largest_square_in_contour(contour, image.shape[:2])
                if points is None:
                    continue
            else:
                # Get convex hull and approximate
                hull = cv2.convexHull(contour)
                peri = cv2.arcLength(hull, True)

                # Try to approximate to quadrilateral
                for eps in [0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(approx) == 4:
                        break

                if len(approx) != 4:
                    # Use bounding rectangle as fallback
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    approx = box.reshape(4, 1, 2).astype(np.int32)

                points = self._order_quadrilateral_points(approx.reshape(4, 2))

                # If the region is not square, expand it to be square
                # This handles cases where white facelets are cut off
                points = self._expand_to_square(points, width, height)

            squareness = self._calculate_squareness(points)

            if squareness < 0.65:
                continue

            # Check saturation within the potential cube region
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)
            region_sat = saturation[mask > 0]
            avg_sat = np.mean(region_sat) if len(region_sat) > 0 else 0

            # Calculate effective area of this quad
            quad_area = cv2.contourArea(points.astype(np.int32))

            # Score based on area, squareness, and color saturation
            score = (quad_area / max_area) * (squareness ** 2) * (avg_sat / 255.0 + 0.5)

            if score > best_score:
                best_score = score
                best_quad = Quadrilateral(points)

        return best_quad

    def _find_largest_square_in_contour(self, contour: np.ndarray, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Find the largest square region that fits inside a contour.

        This is useful when the contour includes extra "tails" from partial
        views of adjacent cube faces.

        Args:
            contour: The contour to search within
            img_shape: Image shape (height, width)

        Returns:
            Quadrilateral points (TL, TR, BR, BL) or None if not found
        """
        # Get bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(contour)

        # Create a mask for this contour
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Find the center of mass of the contour - the cube face is likely centered here
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # The cube face should be roughly square
        # Use a smarter approach: find the largest square centered near the center of mass
        max_size = min(w, h)
        min_valid_size = int(max_size * 0.6)

        best_square = None
        best_score = 0

        # Coarse search first with large steps
        step_size = max(20, max_size // 20)
        size_step = max(20, max_size // 10)

        for size in range(max_size, min_valid_size, -size_step):
            half = size // 2

            # Search in a region around the center of mass
            search_range = max(50, size // 4)
            for dy in range(-search_range, search_range + 1, step_size):
                for dx in range(-search_range, search_range + 1, step_size):
                    test_x = cx - half + dx
                    test_y = cy - half + dy

                    # Bounds check
                    if test_x < 0 or test_y < 0:
                        continue
                    if test_x + size > img_shape[1] or test_y + size > img_shape[0]:
                        continue

                    # Check coverage using ROI instead of full mask operation
                    roi = mask[test_y:test_y+size, test_x:test_x+size]
                    coverage = np.mean(roi) / 255.0

                    # Score: prefer larger squares with good coverage
                    if coverage > 0.85:
                        score = (size / max_size) * coverage
                        if score > best_score:
                            best_score = score
                            best_square = np.array([
                                [test_x, test_y],
                                [test_x + size, test_y],
                                [test_x + size, test_y + size],
                                [test_x, test_y + size]
                            ], dtype=np.float32)

        return best_square

    def _expand_to_square(self, points: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """
        Expand a quadrilateral to be more square.

        If the detected region is rectangular (e.g., missing white facelets),
        expand it in the shorter dimension to make it square.

        Args:
            points: Quadrilateral corners (TL, TR, BR, BL)
            img_width: Image width (for bounds checking)
            img_height: Image height (for bounds checking)

        Returns:
            Expanded quadrilateral points
        """
        # Calculate current dimensions
        top_width = np.linalg.norm(points[1] - points[0])
        bottom_width = np.linalg.norm(points[2] - points[3])
        left_height = np.linalg.norm(points[3] - points[0])
        right_height = np.linalg.norm(points[2] - points[1])

        avg_width = (top_width + bottom_width) / 2
        avg_height = (left_height + right_height) / 2

        # If already roughly square (within 15%), return as-is
        aspect_ratio = min(avg_width, avg_height) / max(avg_width, avg_height)
        if aspect_ratio > 0.85:
            return points

        # Calculate how much to expand
        target_size = max(avg_width, avg_height)
        points = points.copy()

        if avg_width < avg_height:
            # Need to expand horizontally
            expand_amount = (target_size - avg_width) / 2

            # Determine which side to expand (check which side has more room)
            left_x = min(points[0, 0], points[3, 0])
            right_x = max(points[1, 0], points[2, 0])

            left_room = left_x
            right_room = img_width - right_x

            if left_room > right_room:
                # Expand left
                points[0, 0] -= expand_amount
                points[3, 0] -= expand_amount
            else:
                # Expand right
                points[1, 0] += expand_amount
                points[2, 0] += expand_amount

        else:
            # Need to expand vertically
            expand_amount = (target_size - avg_height) / 2

            top_y = min(points[0, 1], points[1, 1])
            bottom_y = max(points[2, 1], points[3, 1])

            top_room = top_y
            bottom_room = img_height - bottom_y

            if top_room > bottom_room:
                # Expand up
                points[0, 1] -= expand_amount
                points[1, 1] -= expand_amount
            else:
                # Expand down
                points[2, 1] += expand_amount
                points[3, 1] += expand_amount

        # Clamp to image bounds
        points[:, 0] = np.clip(points[:, 0], 0, img_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, img_height - 1)

        return points

    def _detect_by_grid_lines(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the cube face by finding the 3x3 grid lines.

        Uses Hough line detection to find horizontal and vertical lines,
        then identifies sets of 4 evenly-spaced lines that form a grid.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=50,
                                minLineLength=min(width, height) // 6,
                                maxLineGap=30)

        if lines is None or len(lines) < 6:
            return None

        # Separate horizontal and vertical lines
        horizontal = []
        vertical = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if angle < 15 or angle > 165:  # Horizontal
                y_avg = (y1 + y2) / 2
                horizontal.append((y_avg, length, line[0]))
            elif 75 < angle < 105:  # Vertical
                x_avg = (x1 + x2) / 2
                vertical.append((x_avg, length, line[0]))

        # Cluster lines and find grid
        h_clusters = self._cluster_lines(horizontal, min_gap=height // 12)
        v_clusters = self._cluster_lines(vertical, min_gap=width // 12)

        if len(h_clusters) < 4 or len(v_clusters) < 4:
            return None

        # Find best 4 evenly-spaced lines in each direction
        h_grid = self._find_evenly_spaced_lines(h_clusters, height)
        v_grid = self._find_evenly_spaced_lines(v_clusters, width)

        if h_grid is None or v_grid is None:
            return None

        # Extract the region defined by the grid
        x_min, x_max = int(v_grid[0]), int(v_grid[3])
        y_min, y_max = int(h_grid[0]), int(h_grid[3])

        # Add small margin
        margin = int(min(x_max - x_min, y_max - y_min) * 0.02)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)

        # Make it square
        w = x_max - x_min
        h = y_max - y_min
        size = max(w, h)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        x_min = max(0, cx - size // 2)
        y_min = max(0, cy - size // 2)
        x_max = min(width, x_min + size)
        y_max = min(height, y_min + size)

        region = image[y_min:y_max, x_min:x_max]

        # Resize to standard square
        if region.shape[0] != region.shape[1]:
            size = min(region.shape[0], region.shape[1])
            region = region[:size, :size]

        return region

    def _fallback_center_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback method: crop center region of the image.

        Uses color saturation to find the best centered square region.
        """
        height, width = image.shape[:2]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Try different crop sizes and find the most colorful one
        best_region = None
        best_score = 0

        for size_pct in [0.6, 0.5, 0.4, 0.7]:
            size = int(min(width, height) * size_pct)

            # Search in a grid pattern
            step = size // 4
            for x in range(0, width - size, step):
                for y in range(0, height - size, step):
                    region_sat = saturation[y:y+size, x:x+size]
                    avg_sat = np.mean(region_sat)

                    # Prefer centered regions
                    cx, cy = x + size // 2, y + size // 2
                    center_dist = np.sqrt((cx - width/2)**2 + (cy - height/2)**2)
                    max_dist = np.sqrt((width/2)**2 + (height/2)**2)
                    center_score = 1.0 - (center_dist / max_dist)

                    score = avg_sat * (0.5 + 0.5 * center_score) * (size_pct + 0.5)

                    if score > best_score:
                        best_score = score
                        best_region = image[y:y+size, x:x+size].copy()

        if best_region is None:
            # Ultimate fallback: center square
            size = int(min(width, height) * 0.6)
            x = (width - size) // 2
            y = (height - size) // 2
            best_region = image[y:y+size, x:x+size].copy()

        return best_region

    def _extract_perspective_corrected(self, image: np.ndarray, quad: Quadrilateral) -> np.ndarray:
        """
        Extract the cube face region with perspective correction.

        Transforms the quadrilateral to a square using homography.
        """
        # Determine output size based on quadrilateral dimensions
        widths = [
            np.linalg.norm(quad.points[1] - quad.points[0]),  # Top edge
            np.linalg.norm(quad.points[2] - quad.points[3])   # Bottom edge
        ]
        heights = [
            np.linalg.norm(quad.points[3] - quad.points[0]),  # Left edge
            np.linalg.norm(quad.points[2] - quad.points[1])   # Right edge
        ]

        output_size = int(max(max(widths), max(heights)))

        # Destination points (square)
        dst_points = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)

        src_points = quad.points.astype(np.float32)

        # Compute homography
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transformation
        warped = cv2.warpPerspective(image, M, (output_size, output_size),
                                      borderMode=cv2.BORDER_REPLICATE)

        return warped

    def _extract_bbox_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract region from bounding box, handling rotation if present.
        """
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        img_h, img_w = image.shape[:2]

        if abs(bbox.rotation) > 0.5:
            # Use perspective transform for rotated bbox
            cx, cy = x + w / 2, y + h / 2
            angle_rad = np.radians(bbox.rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            hw, hh = w / 2, h / 2
            corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_corners = corners @ rotation_matrix.T + np.array([cx, cy])

            quad = Quadrilateral(rotated_corners.astype(np.float32))
            return self._extract_perspective_corrected(image, quad)

        # Simple extraction for axis-aligned bbox
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        region = image[y:y+h, x:x+w]

        # Make square
        min_dim = min(region.shape[0], region.shape[1])
        start_y = (region.shape[0] - min_dim) // 2
        start_x = (region.shape[1] - min_dim) // 2

        return region[start_y:start_y+min_dim, start_x:start_x+min_dim]

    def _trim_black_margin(self, region: np.ndarray) -> np.ndarray:
        """
        Trim black margins around the cube in the extracted region.

        Sometimes the perspective correction includes extra black space
        around the actual cube face. This method finds and removes it.
        """
        if region is None or region.size == 0:
            return region

        height, width = region.shape[:2]

        # Convert to grayscale and find non-black regions
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Use color saturation to find actual cube content (not just brightness)
        # This distinguishes colored facelets from black background
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Combine: either saturated (colored) OR bright (white facelets)
        _, sat_mask = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY)
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        content_mask = cv2.bitwise_or(sat_mask, bright_mask)

        # Find bounding box of non-black content
        coords = cv2.findNonZero(content_mask)
        if coords is None:
            return region

        x, y, w, h = cv2.boundingRect(coords)

        # Only trim if the margin is significant (> 5% on any side)
        margin_threshold = min(width, height) * 0.05
        left_margin = x
        top_margin = y
        right_margin = width - (x + w)
        bottom_margin = height - (y + h)

        if (left_margin > margin_threshold or top_margin > margin_threshold or
            right_margin > margin_threshold or bottom_margin > margin_threshold):

            # Add small padding to not cut into the cube
            pad = int(min(width, height) * 0.02)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(width - x, w + 2 * pad)
            h = min(height - y, h + 2 * pad)

            # Make it square (take the larger dimension)
            size = max(w, h)
            cx, cy = x + w // 2, y + h // 2
            x = max(0, cx - size // 2)
            y = max(0, cy - size // 2)
            x = min(x, width - size)
            y = min(y, height - size)

            if x >= 0 and y >= 0 and x + size <= width and y + size <= height:
                return region[y:y+size, x:x+size]

        return region

    def _extract_facelets_smart(self, face_region: np.ndarray) -> List[np.ndarray]:
        """
        Extract 9 facelets using smart boundary detection.

        Uses color boundaries (black plastic between facelets) to find
        optimal extraction regions.
        """
        # First, trim any black margin around the cube
        face_region = self._trim_black_margin(face_region)

        height, width = face_region.shape[:2]

        # Convert to LAB for better color boundary detection
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0]

        # Find dark regions (black plastic borders)
        _, dark_mask = cv2.threshold(lightness, 50, 255, cv2.THRESH_BINARY_INV)

        # Find grid lines from dark regions
        h_lines = self._find_grid_lines_from_mask(dark_mask, 'horizontal', height)
        v_lines = self._find_grid_lines_from_mask(dark_mask, 'vertical', width)

        # If grid detection fails, use uniform splitting
        if h_lines is None or v_lines is None:
            return self._split_uniform(face_region)

        # Extract facelets based on detected grid
        facelets = []
        for row in range(3):
            for col in range(3):
                y1 = int(h_lines[row])
                y2 = int(h_lines[row + 1])
                x1 = int(v_lines[col])
                x2 = int(v_lines[col + 1])

                # Add margin to avoid black borders
                margin_h = int((y2 - y1) * 0.08)
                margin_w = int((x2 - x1) * 0.08)

                y1 = min(y1 + margin_h, y2 - 1)
                y2 = max(y2 - margin_h, y1 + 1)
                x1 = min(x1 + margin_w, x2 - 1)
                x2 = max(x2 - margin_w, x1 + 1)

                facelet = face_region[y1:y2, x1:x2].copy()
                facelets.append(facelet)

        return facelets

    def _find_grid_lines_from_mask(
        self,
        dark_mask: np.ndarray,
        direction: str,
        dimension: int
    ) -> Optional[List[float]]:
        """
        Find grid lines from dark pixel mask using projection analysis.
        """
        if direction == 'horizontal':
            projection = np.sum(dark_mask, axis=1).astype(float)
        else:
            projection = np.sum(dark_mask, axis=0).astype(float)

        # Smooth projection
        kernel_size = dimension // 30
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            projection = np.convolve(projection, kernel, mode='same')

        # Find peaks (dark line positions)
        threshold = np.mean(projection) + 0.5 * np.std(projection)
        peaks = []

        in_peak = False
        peak_start = 0

        for i, val in enumerate(projection):
            if val > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif val <= threshold and in_peak:
                in_peak = False
                peak_center = (peak_start + i) // 2
                peaks.append(peak_center)

        # We need 2 internal lines + 2 boundaries = 4 lines total
        # Add boundaries
        peaks = [0] + peaks + [dimension]

        # Try to find evenly spaced subset
        if len(peaks) >= 4:
            return self._find_evenly_spaced_lines(peaks, dimension)

        return None

    def _split_uniform(self, face_region: np.ndarray) -> List[np.ndarray]:
        """
        Fallback: split face region uniformly into 9 facelets.
        """
        height, width = face_region.shape[:2]
        facelet_h = height // 3
        facelet_w = width // 3

        # Margin to avoid borders
        margin_h = int(facelet_h * 0.08)
        margin_w = int(facelet_w * 0.08)

        facelets = []
        for row in range(3):
            for col in range(3):
                y_start = row * facelet_h
                y_end = (row + 1) * facelet_h if row < 2 else height
                x_start = col * facelet_w
                x_end = (col + 1) * facelet_w if col < 2 else width

                # Apply margins
                top_margin = margin_h if row > 0 else margin_h // 2
                bottom_margin = margin_h if row < 2 else margin_h // 2
                left_margin = margin_w if col > 0 else margin_w // 2
                right_margin = margin_w if col < 2 else margin_w // 2

                y_start = min(y_start + top_margin, y_end - 1)
                y_end = max(y_end - bottom_margin, y_start + 1)
                x_start = min(x_start + left_margin, x_end - 1)
                x_end = max(x_end - right_margin, x_start + 1)

                facelet = face_region[y_start:y_end, x_start:x_end].copy()
                facelets.append(facelet)

        return facelets

    def _order_quadrilateral_points(self, points: np.ndarray) -> np.ndarray:
        """
        Order quadrilateral points as: top-left, top-right, bottom-right, bottom-left.
        """
        # Sort by y-coordinate first (top vs bottom)
        sorted_by_y = points[np.argsort(points[:, 1])]

        # Top two points
        top_points = sorted_by_y[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]  # Left to right

        # Bottom two points
        bottom_points = sorted_by_y[2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # Left to right

        return np.array([
            top_points[0],      # Top-left
            top_points[1],      # Top-right
            bottom_points[1],   # Bottom-right
            bottom_points[0]    # Bottom-left
        ], dtype=np.float32)

    def _calculate_squareness(self, points: np.ndarray) -> float:
        """
        Calculate how square-like a quadrilateral is (0 to 1, 1 = perfect square).
        """
        # Calculate edge lengths
        edges = [
            np.linalg.norm(points[1] - points[0]),  # Top
            np.linalg.norm(points[2] - points[1]),  # Right
            np.linalg.norm(points[3] - points[2]),  # Bottom
            np.linalg.norm(points[0] - points[3])   # Left
        ]

        # Aspect ratio
        avg_width = (edges[0] + edges[2]) / 2
        avg_height = (edges[1] + edges[3]) / 2
        aspect_ratio = min(avg_width, avg_height) / max(avg_width, avg_height) if max(avg_width, avg_height) > 0 else 0

        # Edge length consistency
        avg_edge = np.mean(edges)
        edge_variance = np.std(edges) / avg_edge if avg_edge > 0 else 1
        edge_consistency = max(0, 1 - edge_variance)

        # Calculate angles
        def angle_at_vertex(p1, vertex, p2):
            v1 = p1 - vertex
            v2 = p2 - vertex
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

        angles = [
            angle_at_vertex(points[3], points[0], points[1]),
            angle_at_vertex(points[0], points[1], points[2]),
            angle_at_vertex(points[1], points[2], points[3]),
            angle_at_vertex(points[2], points[3], points[0])
        ]

        # Perfect square has 90-degree angles
        angle_variance = np.std([abs(a - 90) for a in angles])
        angle_score = max(0, 1 - angle_variance / 45)

        return (aspect_ratio + edge_consistency + angle_score) / 3

    def _cluster_lines(
        self,
        lines: List[Tuple[float, float, np.ndarray]],
        min_gap: int
    ) -> List[float]:
        """
        Cluster nearby lines and return cluster centers.
        """
        if not lines:
            return []

        # Sort by position
        sorted_lines = sorted(lines, key=lambda x: x[0])

        clusters = []
        current_cluster = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            if line[0] - current_cluster[-1][0] < min_gap:
                current_cluster.append(line)
            else:
                # Weighted average by line length
                total_weight = sum(l[1] for l in current_cluster)
                if total_weight > 0:
                    center = sum(l[0] * l[1] for l in current_cluster) / total_weight
                else:
                    center = np.mean([l[0] for l in current_cluster])
                clusters.append(center)
                current_cluster = [line]

        # Don't forget the last cluster
        if current_cluster:
            total_weight = sum(l[1] for l in current_cluster)
            if total_weight > 0:
                center = sum(l[0] * l[1] for l in current_cluster) / total_weight
            else:
                center = np.mean([l[0] for l in current_cluster])
            clusters.append(center)

        return clusters

    def _find_evenly_spaced_lines(
        self,
        positions: List[float],
        max_dim: int
    ) -> Optional[List[float]]:
        """
        Find 4 evenly-spaced lines from a list of candidate positions.

        Returns positions that form a 3x3 grid (4 lines in each direction).
        """
        if len(positions) < 4:
            return None

        positions = sorted(positions)

        best_lines = None
        best_score = float('inf')

        # Try all combinations of 4 lines
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for m in range(k + 1, n):
                        lines = [positions[i], positions[j], positions[k], positions[m]]

                        # Calculate spacings
                        spacings = [lines[1] - lines[0],
                                   lines[2] - lines[1],
                                   lines[3] - lines[2]]

                        avg_spacing = np.mean(spacings)

                        # Skip if spacing is too small or too large
                        if avg_spacing < max_dim * 0.15 or avg_spacing > max_dim * 0.5:
                            continue

                        # Score based on spacing variance (lower = more even)
                        variance = np.var(spacings)
                        uniformity_score = variance / (avg_spacing ** 2) if avg_spacing > 0 else float('inf')

                        if uniformity_score < best_score:
                            best_score = uniformity_score
                            best_lines = lines

        # Threshold for acceptable uniformity
        if best_lines is not None and best_score < 0.1:
            return best_lines

        return None

    # Convenience methods for compatibility with v1 interface

    def segment_from_file(
        self,
        image_path: str,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Convenience method to segment directly from an image file.
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


# Functional interface for compatibility with v1
def segment_cube_face_v2(
    image: np.ndarray,
    bbox: Optional[BoundingBox] = None,
    output_size: int = 64
) -> np.ndarray:
    """
    Functional interface for v2 facelet segmentation.

    Args:
        image: Input image containing a Rubik's cube face (BGR format)
        bbox: Optional bounding box for the face region
        output_size: Size of output facelet images (default 64)

    Returns:
        numpy array of shape (3, 3, output_size, output_size, 3)
    """
    segmenter = FaceletSegmenterV2(output_size=output_size)
    return segmenter.segment(image, bbox)


# Alias for main class name compatibility
FaceletSegmenter = FaceletSegmenterV2
