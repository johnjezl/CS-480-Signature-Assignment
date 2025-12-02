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
    rotation: float = 0.0  # Rotation angle in degrees (counter-clockwise)


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

        # Only apply secondary rotation correction if bbox didn't already have rotation
        if abs(bbox.rotation) < 0.5:
            face_region = self._correct_rotation(face_region)

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

        # Threshold saturation to find colorful areas (lower threshold for white/yellow)
        _, sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

        # Use coarse edge detection to find cube boundary (not fine details like logos)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Heavy blur to smooth out logos and internal details
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Detect only strong edges with high thresholds (outer cube boundaries)
        edges = cv2.Canny(blurred, 60, 180)

        # Morphological operations to connect edges and fill the cube region
        kernel = np.ones((7, 7), np.uint8)
        edges_filled = cv2.dilate(edges, kernel, iterations=4)
        edges_filled = cv2.morphologyEx(edges_filled, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Combine saturation (for color) and filled edges (for structure)
        # Prioritize saturation to avoid false detections from logos
        combined_mask = cv2.addWeighted(sat_mask, 0.7, edges_filled, 0.3, 0)
        _, combined_mask = cv2.threshold(combined_mask, 100, 255, cv2.THRESH_BINARY)

        # Final cleanup: fill holes inside the cube region
        kernel_large = np.ones((11, 11), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours in combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = 0

        # Expected cube size range (as fraction of image dimensions)
        # Cube must be reasonably large to avoid detecting individual facelets or logos
        min_size = min(width, height) * 0.15  # At least 15% of smallest dimension
        max_size = min(width, height) * 0.70  # At most 70% - larger usually means merged with background

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size * min_size:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate squareness and saturation first (needed for size filtering)
            squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            roi_sat = saturation[y:y+h, x:x+w]
            avg_saturation = roi_sat.mean()

            # Reject low-saturation regions - probably not a colorful cube
            if avg_saturation < 60:
                continue

            # Filter by size - but allow large square-ish colored regions
            # (may be cube merged with similar background, we'll refine later)
            too_small = w < min_size or h < min_size
            too_large = w > max_size or h > max_size
            # Relaxed: squareness > 0.85 (was 0.9) to allow slightly non-square merged regions
            is_square_colored = squareness > 0.85 and avg_saturation > 60

            if too_small:
                continue
            if too_large and not is_square_colored:
                continue

            # Check proximity to image edges
            margin_x = int(width * 0.02)
            margin_y = int(height * 0.02)

            edges_touched = 0
            if x < margin_x: edges_touched += 1
            if y < margin_y: edges_touched += 1
            if x + w > width - margin_x: edges_touched += 1
            if y + h > height - margin_y: edges_touched += 1

            # Reject if touching 2+ edges, UNLESS it has very strong color saturation
            # OR is square-ish and large (likely cube merged with similar background)
            if edges_touched >= 2:
                # Allow if:
                # 1. Touching exactly 2 edges AND has strong colors (sat > 80), OR
                # 2. Square-ish (>0.85) with decent saturation (>60) - may contain cube
                is_strong_colored = edges_touched == 2 and avg_saturation > 80

                if not (is_strong_colored or is_square_colored):
                    continue

            if squareness < 0.75:  # Cube faces must be quite square
                continue

            # Score based on size, squareness, and color saturation
            # (avg_saturation already calculated above for edge check)

            # Prefer medium-sized, square regions - saturation helps but isn't critical
            # (cubes with white/gray facelets have low saturation but correct shape)
            size_score = (w * h) / (max_size * max_size)  # Normalize by max size

            # Bonus for being very square (0.95+)
            squareness_bonus = 1.5 if squareness > 0.95 else 1.0

            # Use sqrt of saturation to reduce its weight - shape matters more than color
            sat_factor = np.sqrt(avg_saturation / 255.0) * 0.5 + 0.5  # Range 0.5-1.0

            score = (squareness ** 2) * size_score * sat_factor * squareness_bonus

            if score > best_score:
                best_score = score
                best_bbox = BoundingBox(x, y, w, h)

        # If we found a bbox, try grid-based refinement within the color region
        # This helps when the color-based detection is approximate
        if best_bbox is not None and max(width, height) > 1500:
            # Try grid detection within this color-based region
            grid_bbox = self._detect_grid_within_region(image, best_bbox)
            if grid_bbox is not None:
                best_bbox = grid_bbox

            # Check if we might be cutting off edges (low contrast facelets)
            extrapolated_bbox = self._extrapolate_from_top_rows(image, best_bbox, saturation)
            if extrapolated_bbox is not None:
                best_bbox = extrapolated_bbox
            else:
                # Fallback to saturation-based refinement for merged regions
                bbox_size = max(best_bbox.width, best_bbox.height)
                image_size = max(width, height)

                if bbox_size > image_size * 0.7:
                    refined_bbox = self._refine_large_region(image, best_bbox, saturation)
                    if refined_bbox is not None:
                        best_bbox = refined_bbox

                        # Try extrapolation again on the refined bbox
                        extrapolated_bbox = self._extrapolate_from_top_rows(image, best_bbox, saturation)
                        if extrapolated_bbox is not None:
                            best_bbox = extrapolated_bbox

        # Fallback: use centered square region
        if best_bbox is None:
            # Check if the image is already mostly cube (pre-cropped/bounded)
            # If image is square-ish and reasonably sized, use most of it
            image_squareness = min(width, height) / max(width, height)
            image_is_small = max(width, height) < 1500  # Likely a cropped image

            if image_squareness > 0.8 and image_is_small:
                # Image is already cropped to cube - use 90% of it with margins
                size = int(min(width, height) * 0.9)
                x = (width - size) // 2
                y = (height - size) // 2
                best_bbox = BoundingBox(x, y, size, size)
            else:
                # Original large image - scan for high-saturation region
                # Try multiple sizes since cube size varies
                best_bbox = None
                best_score = 0

                for size_pct in [0.35, 0.45, 0.55, 0.65]:  # Try 35%, 45%, 55%, 65% of image
                    size = int(min(width, height) * size_pct)
                    step = size // 6

                    for test_x in range(0, width - size, step):
                        for test_y in range(int(height * 0.15), int(height * 0.85) - size, step):
                            region_sat = saturation[test_y:test_y+size, test_x:test_x+size]
                            avg_sat = region_sat.mean() if region_sat.size > 0 else 0
                            # Score: saturation with size bonus (larger regions more likely to be full cube)
                            # This prevents small high-saturation corners from winning over full cube
                            score = avg_sat * (1 + size_pct * 0.5)
                            if score > best_score:
                                best_score = score
                                best_bbox = BoundingBox(test_x, test_y, size, size)

                # Fallback to center if nothing found
                if best_bbox is None:
                    size = int(min(width, height) * 0.5)
                    best_bbox = BoundingBox((width - size) // 2, (height - size) // 2, size, size)

        # Ensure final bbox is square
        if best_bbox is not None and best_bbox.width != best_bbox.height:
            size = max(best_bbox.width, best_bbox.height)
            # Center the square on the original bbox
            x = best_bbox.x - (size - best_bbox.width) // 2
            y = best_bbox.y - (size - best_bbox.height) // 2
            # Keep in bounds
            x = max(0, min(x, width - size))
            y = max(0, min(y, height - size))
            size = min(size, width - x, height - y)
            best_bbox = BoundingBox(x, y, size, size)

        # Detect rotation angle for the bbox
        if best_bbox is not None:
            rotation = self._detect_rotation_angle(image, best_bbox)
            if abs(rotation) > 0.5:
                # Expand bbox to contain the rotated cube
                # For a square rotated by θ, axis-aligned bbox must be expanded by (|cos θ| + |sin θ|)
                # Reduce by 5% since the theoretical max is rarely needed in practice
                angle_rad = np.radians(abs(rotation))
                expansion_factor = (abs(np.cos(angle_rad)) + abs(np.sin(angle_rad))) * 0.90

                original_size = best_bbox.width
                expanded_size = int(original_size * expansion_factor)

                # Center the expanded bbox on the original
                cx = best_bbox.x + original_size // 2
                cy = best_bbox.y + original_size // 2
                new_x = max(0, cx - expanded_size // 2)
                new_y = max(0, cy - expanded_size // 2)

                # Keep in bounds
                if new_x + expanded_size > width:
                    new_x = width - expanded_size
                if new_y + expanded_size > height:
                    new_y = height - expanded_size

                best_bbox = BoundingBox(new_x, new_y, expanded_size, expanded_size, rotation)
                print(f"     [BBox] Set rotation: {rotation:.2f}°, expanded {original_size} -> {expanded_size}")

        return best_bbox

    def _expand_for_bottom_facelets(self, image: np.ndarray, bbox: BoundingBox, saturation: np.ndarray) -> BoundingBox:
        """
        Expand bounding box downward to capture low-saturation bottom facelets.

        White facelets on white background have low saturation and may get cut off.
        This expands the bbox to ensure we capture the full bottom row.

        Args:
            image: Input image
            bbox: Current bounding box
            saturation: Saturation channel from HSV

        Returns:
            Expanded BoundingBox
        """
        img_h, img_w = image.shape[:2]

        # Expand downward by ~20% of current height
        expansion = int(bbox.height * 0.2)

        # Also expand left/right slightly for symmetry
        side_expansion = int(bbox.width * 0.1)

        new_x = max(0, bbox.x - side_expansion)
        new_y = bbox.y  # Don't move top edge
        new_width = min(bbox.width + 2 * side_expansion, img_w - new_x)
        new_height = min(bbox.height + expansion, img_h - new_y)

        # Make it square
        size = max(new_width, new_height)
        cx = new_x + new_width // 2
        cy = new_y + new_height // 2

        x_sq = max(0, cx - size // 2)
        y_sq = max(0, cy - size // 2)

        # Ensure it fits in image
        if x_sq + size > img_w:
            x_sq = img_w - size
        if y_sq + size > img_h:
            y_sq = img_h - size

        return BoundingBox(x_sq, y_sq, size, size)

    def _extrapolate_from_top_rows(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        saturation: np.ndarray
    ) -> Optional[BoundingBox]:
        """
        Expand bbox to account for low-contrast edges (white facelets on white background).

        When the grid detection cuts off parts of the cube due to low contrast,
        expand the bbox by ~15% in all directions to capture the full cube.

        Args:
            image: Input image
            bbox: Current bounding box from grid detection
            saturation: Saturation channel from HSV

        Returns:
            Expanded BoundingBox, or None if expansion not needed
        """
        img_h, img_w = image.shape[:2]
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Ensure bounds are valid
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Only try to expand reasonably-sized bboxes (40-75% of image)
        # If bbox is already too large or too small, don't expand
        bbox_size = max(w, h)
        img_size = min(img_w, img_h)
        size_ratio = bbox_size / img_size

        if size_ratio > 0.75 or size_ratio < 0.4:
            return None  # Bbox is too large or too small to expand reliably

        # Check if the bottom portion has low saturation (indicating white facelets)
        bottom_third_y = y + int(h * 0.67)
        bottom_region_sat = saturation[bottom_third_y:y+h, x:x+w]
        bottom_avg_sat = bottom_region_sat.mean() if bottom_region_sat.size > 0 else 100

        # Check left edge saturation - might be cutting off colored facelets
        left_strip_sat = saturation[y:y+h, max(0, x-100):x]
        left_avg_sat = left_strip_sat.mean() if left_strip_sat.size > 0 else 0

        # Check TOP edge saturation - might be cutting off top row
        top_strip_sat = saturation[max(0, y-100):y, x:x+w]
        top_avg_sat = top_strip_sat.mean() if top_strip_sat.size > 0 else 0

        # Determine what expansion is needed
        needs_bottom_expansion = bottom_avg_sat < 50
        needs_left_expansion = left_avg_sat > 40
        needs_top_expansion = top_avg_sat > 40

        if not (needs_bottom_expansion or needs_left_expansion or needs_top_expansion):
            return None

        # Calculate directional expansions - only expand where needed
        base_expansion = int(max(w, h) * 0.15)

        top_expansion = int(base_expansion * 0.60) if needs_top_expansion else 0
        bottom_expansion = int(base_expansion * 0.60) if needs_bottom_expansion else 0
        left_expansion = int(base_expansion * 0.60) if needs_left_expansion else 0

        new_x = max(0, x - left_expansion)
        new_y = max(0, y - top_expansion)
        new_size = max(w, h) + top_expansion + bottom_expansion

        # Make it square - use the larger dimension
        new_size = max(new_size, w + left_expansion, h + top_expansion + bottom_expansion)

        # Ensure it fits in image
        if new_x + new_size > img_w:
            new_x = max(0, img_w - new_size)
        if new_y + new_size > img_h:
            new_y = max(0, img_h - new_size)
            new_size = min(new_size, img_h - new_y)

        # Only return if this would actually change the bbox significantly
        if abs(new_x - x) < 50 and abs(new_size - max(w, h)) < 50:
            return None

        return BoundingBox(int(new_x), int(new_y), int(new_size), int(new_size))

    def _detect_grid_within_region(self, image: np.ndarray, color_bbox: BoundingBox) -> Optional[BoundingBox]:
        """
        Detect 3x3 grid pattern within a color-detected region.

        This refines the color-based detection by finding the actual grid lines.

        Args:
            image: Full input image
            color_bbox: Bounding box from color-based detection

        Returns:
            Refined BoundingBox based on grid, or None if grid not found
        """
        # Extract the color-detected region
        x, y, w, h = color_bbox.x, color_bbox.y, color_bbox.width, color_bbox.height

        # Ensure bounds are valid
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        region = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 30, 100)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=min(w, h)//4, maxLineGap=30)

        if lines is None:
            return None

        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 10 or angle > 170:
                horizontal_lines.append((x1, y1, x2, y2))
            elif 80 < angle < 100:
                vertical_lines.append((x1, y1, x2, y2))

        # Cluster lines
        h_clusters = self._cluster_lines_simple(horizontal_lines, coord_idx=1)
        v_clusters = self._cluster_lines_simple(vertical_lines, coord_idx=0)

        if len(h_clusters) < 4 or len(v_clusters) < 4:
            return None

        # Find 3x3 grid
        h_grid = self._find_3x3_grid_in_region(h_clusters, h)
        v_grid = self._find_3x3_grid_in_region(v_clusters, w)

        if h_grid is None or v_grid is None:
            return None

        # Convert region coordinates back to image coordinates
        x_min = int(v_grid[0]) + x
        x_max = int(v_grid[3]) + x
        y_min = int(h_grid[0]) + y
        y_max = int(h_grid[3]) + y

        grid_w = x_max - x_min
        grid_h = y_max - y_min

        # Make it square
        size = max(grid_w, grid_h)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        x_sq = cx - size // 2
        y_sq = cy - size // 2

        return BoundingBox(x_sq, y_sq, size, size)

    def _cluster_lines_simple(self, lines: List[Tuple[int, int, int, int]], coord_idx: int) -> List[float]:
        """Cluster parallel lines by position."""
        if not lines:
            return []

        coords = []
        for x1, y1, x2, y2 in lines:
            coord = (x1 + x2) / 2 if coord_idx == 0 else (y1 + y2) / 2
            coords.append(coord)

        coords = sorted(coords)
        clusters = []
        current = [coords[0]]

        for coord in coords[1:]:
            if coord - current[-1] < 50:  # Tighter clustering for region
                current.append(coord)
            else:
                clusters.append(np.mean(current))
                current = [coord]

        if current:
            clusters.append(np.mean(current))

        return clusters

    def _cluster_lines_extrapolation(self, lines: List[Tuple[int, int, int, int]], coord_idx: int) -> List[float]:
        """Cluster parallel lines by position with more lenient tolerance for extrapolation."""
        if not lines:
            return []

        coords = []
        for x1, y1, x2, y2 in lines:
            coord = (x1 + x2) / 2 if coord_idx == 0 else (y1 + y2) / 2
            coords.append(coord)

        coords = sorted(coords)
        clusters = []
        current = [coords[0]]

        # Use 80px tolerance instead of 50px for more lenient clustering
        for coord in coords[1:]:
            if coord - current[-1] < 80:
                current.append(coord)
            else:
                clusters.append(np.mean(current))
                current = [coord]

        if current:
            clusters.append(np.mean(current))

        return clusters

    def _find_3x3_grid_in_region(self, clusters: List[float], max_dim: int) -> Optional[List[float]]:
        """Find 4 evenly-spaced lines forming a 3x3 grid within a region."""
        if len(clusters) < 4:
            return None

        best_grid = None
        best_score = float('inf')

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                for k in range(j+1, len(clusters)):
                    for m in range(k+1, len(clusters)):
                        lines = [clusters[i], clusters[j], clusters[k], clusters[m]]

                        spacing = [lines[1]-lines[0], lines[2]-lines[1], lines[3]-lines[2]]
                        avg_spacing = np.mean(spacing)

                        if avg_spacing < 50:  # Too small
                            continue

                        total_size = lines[3] - lines[0]
                        if total_size < max_dim * 0.5 or total_size > max_dim * 1.0:
                            continue

                        variance = np.var(spacing)
                        uniformity = variance / (avg_spacing ** 2) if avg_spacing > 0 else float('inf')

                        if uniformity < best_score:
                            best_score = uniformity
                            best_grid = lines

        return best_grid if best_grid else None

    def _refine_large_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        saturation: np.ndarray
    ) -> Optional[BoundingBox]:
        """
        Refine a large bounding box to find the actual cube within it.

        When the cube's edge blends with background (e.g., white cube on white surface),
        the detection may merge them. This method finds the most colorful square
        sub-region within the large bbox.

        Args:
            image: Input image
            bbox: Large bounding box that may contain cube + background
            saturation: Saturation channel from HSV

        Returns:
            Refined BoundingBox, or None if refinement fails
        """
        # Extract the large region
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        region_sat = saturation[y:y+h, x:x+w]

        # Expected cube size based on image dimensions
        img_h, img_w = image.shape[:2]
        expected_cube_size = int(min(img_w, img_h) * 0.65)  # Cube is roughly 65% of image
        min_cube_size = int(min(img_w, img_h) * 0.5)  # At least 50%
        max_cube_size = int(min(img_w, img_h) * 0.75)  # At most 75%

        # Try different square sizes to find the most saturated region
        best_refined_bbox = None
        best_refined_score = 0

        for cube_size in range(max_cube_size, min_cube_size, -100):
            if cube_size > min(w, h):
                continue

            # Slide window across the region
            step = max(50, cube_size // 10)
            for ry in range(0, h - cube_size + 1, step):
                for rx in range(0, w - cube_size + 1, step):
                    # Extract this square sub-region's saturation
                    sub_sat = region_sat[ry:ry+cube_size, rx:rx+cube_size]
                    avg_sat = sub_sat.mean()

                    # Score based on saturation
                    # Prefer regions closer to expected cube size
                    size_diff = abs(cube_size - expected_cube_size) / expected_cube_size
                    size_penalty = max(0, 1.0 - size_diff)

                    score = avg_sat * size_penalty

                    if score > best_refined_score:
                        best_refined_score = score
                        # Convert back to image coordinates
                        best_refined_bbox = BoundingBox(
                            x + rx,
                            y + ry,
                            cube_size,
                            cube_size
                        )

        # Only return refined bbox if it's significantly better than original
        if best_refined_bbox and best_refined_score > 0:
            return best_refined_bbox

        return None

    def _detect_rotation_angle(self, image: np.ndarray, bbox: BoundingBox) -> float:
        """
        Detect the rotation angle of the cube within the bounding box.

        Args:
            image: Input image
            bbox: Bounding box for the cube region

        Returns:
            Rotation angle in degrees (counter-clockwise positive)
        """
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        img_h, img_w = image.shape[:2]

        # Ensure bounds are valid
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Extract region
        region = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=min(w, h)//4, maxLineGap=20)

        if lines is None or len(lines) < 4:
            return 0.0

        # Collect angles with lengths
        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Normalize to [-45, 45] range
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90

            if abs(angle) < 15:
                line_data.append((angle, length))

        if len(line_data) < 4:
            return 0.0

        # Use longest lines
        line_data.sort(key=lambda x: x[1], reverse=True)
        top_lines = line_data[:max(4, len(line_data) // 4)]

        angles = np.array([d[0] for d in top_lines])
        weights = np.array([d[1] for d in top_lines])

        detected_angle = np.sum(angles * weights) / np.sum(weights)

        # Scale up moderate rotations (grid lines underestimate)
        if 2.0 < abs(detected_angle) < 8.0:
            detected_angle *= 1.75

        return detected_angle

    def _extract_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract and square-crop the face region from the image.

        Handles rotated bounding boxes using perspective transform.

        Args:
            image: Input image
            bbox: Bounding box for extraction (may include rotation)

        Returns:
            Extracted region as numpy array
        """
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        rotation = bbox.rotation

        # Ensure bounds are within image
        img_height, img_width = image.shape[:2]

        # If rotation is significant, use perspective transform
        if abs(rotation) > 0.5:
            # Calculate center of bbox
            cx = x + w / 2
            cy = y + h / 2

            # Calculate rotated corners
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Half dimensions
            hw, hh = w / 2, h / 2

            # Corners relative to center, then rotated
            corners = np.array([
                [-hw, -hh],  # top-left
                [hw, -hh],   # top-right
                [hw, hh],    # bottom-right
                [-hw, hh]    # bottom-left
            ])

            # Rotate corners
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_corners = corners @ rotation_matrix.T

            # Translate to image coordinates
            src_points = rotated_corners + np.array([cx, cy])
            src_points = src_points.astype(np.float32)

            # Destination points (axis-aligned square)
            size = max(w, h)
            dst_points = np.array([
                [0, 0],
                [size, 0],
                [size, size],
                [0, size]
            ], dtype=np.float32)

            # Compute perspective transform
            M = cv2.getPerspectiveTransform(src_points, dst_points)

            # Apply transform
            square_region = cv2.warpPerspective(image, M, (int(size), int(size)),
                                                 borderMode=cv2.BORDER_REPLICATE)

            return square_region

        # No significant rotation - use simple extraction
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        # Extract the region
        region = image[y:y+h, x:x+w]

        # Make it square by cropping to the smaller dimension
        min_dim = min(region.shape[0], region.shape[1])

        # Center crop to square
        start_y = (region.shape[0] - min_dim) // 2
        start_x = (region.shape[1] - min_dim) // 2

        square_region = region[start_y:start_y+min_dim, start_x:start_x+min_dim]

        return square_region

    def _correct_rotation(self, face_region: np.ndarray) -> np.ndarray:
        """
        Detect and correct slight rotation in the cube face image.

        Uses line detection to find the dominant angle of the cube grid lines
        and rotates to align them with the image axes.

        Args:
            face_region: Square image of the cube face

        Returns:
            Rotation-corrected image (or original if no significant rotation detected)
        """
        height, width = face_region.shape[:2]

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=width//4, maxLineGap=20)

        if lines is None or len(lines) < 4:
            return face_region

        # Collect angles from detected lines with their lengths
        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Normalize to [-45, 45] range (we care about deviation from 0 or 90)
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90

            # Only consider small rotations (within ±15 degrees)
            if abs(angle) < 15:
                line_data.append((angle, length))

        if len(line_data) < 4:
            return face_region

        # Sort by length and use only the longest 25% of lines
        line_data.sort(key=lambda x: x[1], reverse=True)
        top_lines = line_data[:max(4, len(line_data) // 4)]

        angles = np.array([d[0] for d in top_lines])
        weights = np.array([d[1] for d in top_lines])

        # Use weighted average of longest lines
        detected_angle = np.sum(angles * weights) / np.sum(weights)

        # The internal grid lines often underestimate the actual cube rotation
        # because they're shorter and affected by facelet borders. Scale up slightly.
        if 2.0 < abs(detected_angle) < 8.0:
            detected_angle *= 2.2  # Scale up moderate rotations

        print(f"     [Rotation] Detected angle: {detected_angle:.2f}° from {len(angles)} lines")

        # Only correct if rotation is significant (> 0.5 degree) but not too large
        if abs(detected_angle) < 0.5 or abs(detected_angle) > 15:
            print(f"     [Rotation] Skipping (threshold: 0.5-15°)")
            return face_region

        print(f"     [Rotation] Applying {detected_angle:.2f}° correction")

        # Rotate around center
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, detected_angle, 1.0)

        # Calculate new bounding box size to avoid cutting corners
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)

        # Adjust rotation matrix for new image size
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2

        # Rotate the image
        rotated = cv2.warpAffine(face_region, rotation_matrix, (new_width, new_height),
                                  borderMode=cv2.BORDER_REPLICATE)

        # Crop back to original size from center
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        cropped = rotated[start_y:start_y+height, start_x:start_x+width]

        return cropped

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
