"""
Rubik's Cube Face Segmentation Component - Version 3

This version uses a contour-based approach to directly detect individual facelets:
1. Find all square-shaped contours in the image
2. Filter contours by size, aspect ratio, and neighbor relationships
3. Validate that detected squares form a 3x3 grid pattern
4. Extract facelets from validated grid positions

This approach is inspired by common Rubik's cube detection algorithms that:
- Use contour detection to find square stickers/facelets
- Validate facelets by checking if they have neighboring facelets
- Reconstruct the 3x3 grid from spatial relationships

Key insight: A valid facelet must have at least one neighboring facelet
(corner facelets have 2 neighbors, edge facelets have 3, center has 4).

Facelet ordering:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    """Represents a bounding box for the cube face region."""
    x: int
    y: int
    width: int
    height: int
    rotation: float = 0.0  # Rotation angle in degrees (counter-clockwise)


@dataclass(eq=False)  # Disable auto-generated __eq__ to avoid numpy array comparison issues
class DetectedFacelet:
    """Represents a detected facelet candidate."""
    contour: np.ndarray
    center: Tuple[float, float]
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    aspect_ratio: float
    approx_poly: np.ndarray  # Approximated polygon points
    dominant_color: Optional[np.ndarray] = None
    neighbor_count: int = 0
    grid_position: Optional[Tuple[int, int]] = None  # (row, col) if assigned


class FaceletSegmenterV3:
    """
    Version 3 of the Rubik's cube face segmenter using contour-based detection.

    This approach directly detects individual square facelets rather than
    first finding the cube boundary and then subdividing.

    Key features:
    - Contour detection to find square-shaped regions
    - Neighbor analysis to validate facelets (real facelets have neighbors)
    - Grid reconstruction from validated facelet positions
    - Dominant color extraction using k-means

    Usage:
        segmenter = FaceletSegmenterV3(output_size=64)
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
                  If provided, detection is limited to this region.

        Returns:
            numpy array of shape (3, 3, 64, 64, 3) representing a 3x3 grid
            Access pattern: [row][col][height][width][channel]
        """
        # If bbox provided, extract that region first
        if bbox is not None:
            image = self._extract_bbox_region(image, bbox)

        # Try direct facelet extraction first (more accurate)
        facelets = self._detect_and_extract_facelets_directly(image)

        if facelets is not None:
            return facelets

        # Fallback: Main detection pipeline (region-based)
        face_region = self._detect_cube_face(image)

        # Split into 9 facelets
        facelets_list = self._extract_facelets(face_region)

        # Resize each facelet to output size
        resized_facelets = [
            cv2.resize(facelet, (self.output_size, self.output_size),
                      interpolation=cv2.INTER_AREA)
            for facelet in facelets_list
        ]

        # Reshape to 3x3 grid
        facelets_array = np.array(resized_facelets, dtype=np.uint8)
        facelets_grid = facelets_array.reshape(3, 3, self.output_size, self.output_size, 3)

        return facelets_grid

    def _detect_and_extract_facelets_directly(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect facelets and extract them directly from their contour positions.

        This method extracts facelets directly from detected contour bounding boxes
        rather than extracting a region and re-splitting it. This produces tighter
        crops around the actual colored facelet areas.

        Returns:
            numpy array of shape (3, 3, output_size, output_size, 3) if successful,
            None if direct extraction fails
        """
        height, width = image.shape[:2]

        if self.debug:
            print(f"[V3-Direct] Processing image: {width}x{height}")

        # Step 1: Find candidate facelets
        candidates = self._find_square_contours(image)

        if self.debug:
            print(f"[V3-Direct] Found {len(candidates)} candidate squares")

        if len(candidates) < 9:
            candidates = self._find_square_contours(image, relaxed=True)
            if self.debug:
                print(f"[V3-Direct] Relaxed search found {len(candidates)} candidates")

        # Step 2: Extract color info
        self._analyze_candidate_colors(image, candidates)

        # Step 3: Filter by color
        color_filtered = self._filter_by_color(candidates)

        if self.debug:
            print(f"[V3-Direct] Color filtering kept {len(color_filtered)} candidates")

        # Expand with nearby candidates if needed
        if 5 <= len(color_filtered) < 9:
            expanded = self._expand_with_nearby_candidates(color_filtered, candidates)
            if len(expanded) >= 9:
                candidates = expanded
            else:
                candidates = color_filtered
        elif len(color_filtered) >= 9:
            candidates = color_filtered

        # Step 4: Find best grid cluster if too many candidates
        if len(candidates) > 9:
            candidates = self._find_best_grid_cluster(candidates, image.shape[:2])

        # Step 5: Validate by neighbors
        validated = self._validate_by_neighbors(candidates)

        if self.debug:
            print(f"[V3-Direct] Validated {len(validated)} facelets")

        # Step 6: Reconstruct grid positions
        grid_facelets = self._reconstruct_grid(validated, image.shape[:2])

        if grid_facelets is None or len(grid_facelets) < 5:
            if self.debug:
                print("[V3-Direct] Could not reconstruct grid (too few facelets), falling back")
            return None

        # Step 7: Build grid from assigned positions
        grid = [[None for _ in range(3)] for _ in range(3)]
        for facelet in grid_facelets:
            if facelet.grid_position is not None:
                row, col = facelet.grid_position
                grid[row][col] = facelet

        # Check if we have all 9 positions filled
        missing = []
        for row in range(3):
            for col in range(3):
                if grid[row][col] is None:
                    missing.append((row, col))

        if missing:
            # Try to infer missing positions
            grid = self._infer_missing_facelet_positions(grid, grid_facelets, image.shape[:2])
            # Re-check
            missing = [(r, c) for r in range(3) for c in range(3) if grid[r][c] is None]

        if missing:
            if self.debug:
                print(f"[V3-Direct] Missing positions: {missing}, falling back")
            return None

        # Step 8: Extract each facelet directly from its detected position
        facelets_grid = np.zeros((3, 3, self.output_size, self.output_size, 3), dtype=np.uint8)

        for row in range(3):
            for col in range(3):
                facelet = grid[row][col]
                x, y, w, h = facelet.bounding_rect

                # Apply inward margin to avoid black borders (15% from each edge)
                margin_x = int(w * 0.15)
                margin_y = int(h * 0.15)

                x1 = max(0, x + margin_x)
                y1 = max(0, y + margin_y)
                x2 = min(image.shape[1], x + w - margin_x)
                y2 = min(image.shape[0], y + h - margin_y)

                # Ensure valid region
                if x2 <= x1 or y2 <= y1:
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                # Extract and resize
                facelet_img = image[y1:y2, x1:x2]
                resized = cv2.resize(facelet_img, (self.output_size, self.output_size),
                                    interpolation=cv2.INTER_AREA)
                facelets_grid[row, col] = resized

        if self.debug:
            print("[V3-Direct] Successfully extracted all 9 facelets directly")

        return facelets_grid

    def _infer_missing_facelet_positions(
        self,
        grid: List[List[Optional[DetectedFacelet]]],
        detected: List[DetectedFacelet],
        image_shape: Tuple[int, int]
    ) -> List[List[Optional[DetectedFacelet]]]:
        """
        Infer positions of missing facelets based on detected ones.

        When we have detected facelets but they don't fill all grid positions,
        estimate where the missing ones should be based on the grid spacing.
        """
        # Get detected positions and their coordinates
        detected_positions = []
        for row in range(3):
            for col in range(3):
                if grid[row][col] is not None:
                    f = grid[row][col]
                    detected_positions.append((row, col, f.center[0], f.center[1], f.bounding_rect))

        if len(detected_positions) < 3:
            return grid

        # Calculate average spacing
        x_coords_by_col = {0: [], 1: [], 2: []}
        y_coords_by_row = {0: [], 1: [], 2: []}

        for row, col, cx, cy, _ in detected_positions:
            x_coords_by_col[col].append(cx)
            y_coords_by_row[row].append(cy)

        # Calculate column x positions
        col_x = {}
        for col in range(3):
            if x_coords_by_col[col]:
                col_x[col] = np.mean(x_coords_by_col[col])

        # Calculate row y positions
        row_y = {}
        for row in range(3):
            if y_coords_by_row[row]:
                row_y[row] = np.mean(y_coords_by_row[row])

        # Get median facelet size for spacing estimation
        sizes = [max(f.bounding_rect[2], f.bounding_rect[3]) for f in detected]
        median_size = int(np.median(sizes))

        # Estimate spacing from positions if we have at least 2 cols/rows
        col_spacing = None
        row_spacing = None

        if len(col_x) >= 2:
            cols = sorted(col_x.keys())
            spacings = []
            for i in range(len(cols) - 1):
                spacings.append((col_x[cols[i+1]] - col_x[cols[i]]) / (cols[i+1] - cols[i]))
            col_spacing = np.mean(spacings)

        if len(row_y) >= 2:
            rows = sorted(row_y.keys())
            spacings = []
            for i in range(len(rows) - 1):
                spacings.append((row_y[rows[i+1]] - row_y[rows[i]]) / (rows[i+1] - rows[i]))
            row_spacing = np.mean(spacings)

        # Default spacing based on facelet size if we couldn't calculate it
        if col_spacing is None:
            col_spacing = median_size * 1.15  # Facelet + small gap
        if row_spacing is None:
            row_spacing = median_size * 1.15

        # Infer missing column positions
        if len(col_x) >= 1:
            ref_col = list(col_x.keys())[0]
            ref_x = col_x[ref_col]
            for col in range(3):
                if col not in col_x:
                    col_x[col] = ref_x + (col - ref_col) * col_spacing

        # Infer missing row positions
        if len(row_y) >= 1:
            ref_row = list(row_y.keys())[0]
            ref_y = row_y[ref_row]
            for row in range(3):
                if row not in row_y:
                    row_y[row] = ref_y + (row - ref_row) * row_spacing

        # Create synthetic facelets for missing positions
        for row in range(3):
            for col in range(3):
                if grid[row][col] is None and col in col_x and row in row_y:
                    cx = col_x[col]
                    cy = row_y[row]
                    x = int(cx - median_size / 2)
                    y = int(cy - median_size / 2)

                    # Create synthetic facelet
                    synthetic = DetectedFacelet(
                        contour=np.array([]),
                        center=(cx, cy),
                        bounding_rect=(x, y, median_size, median_size),
                        area=median_size * median_size,
                        aspect_ratio=1.0,
                        approx_poly=np.array([]),
                        grid_position=(row, col)
                    )
                    grid[row][col] = synthetic

        return grid

    def _detect_cube_face(self, image: np.ndarray) -> np.ndarray:
        """
        Main detection pipeline using contour-based facelet detection.

        Steps:
        1. Find all square-shaped contours
        2. Filter by size and aspect ratio
        3. Extract color info and filter by saturation
        4. Validate by neighbor relationships
        5. Reconstruct grid from validated facelets
        6. Extract the face region
        """
        height, width = image.shape[:2]

        if self.debug:
            print(f"[V3] Processing image: {width}x{height}")

        # Step 1: Find candidate facelets (square contours)
        candidates = self._find_square_contours(image)

        if self.debug:
            print(f"[V3] Found {len(candidates)} candidate squares")

        if len(candidates) < 9:
            # Not enough candidates - try with different parameters
            candidates = self._find_square_contours(image, relaxed=True)
            if self.debug:
                print(f"[V3] Relaxed search found {len(candidates)} candidates")

        # Step 2: Extract color info for each candidate
        self._analyze_candidate_colors(image, candidates)

        # Step 3: Filter by color - keep colorful candidates (cube facelets are saturated)
        color_filtered = self._filter_by_color(candidates)

        if self.debug:
            print(f"[V3] Color filtering kept {len(color_filtered)} of {len(candidates)} candidates")

        # If we have colorful candidates, use them to find nearby low-saturation ones
        # (white facelets might look similar to gray background objects)
        if 5 <= len(color_filtered) < 9:
            expanded = self._expand_with_nearby_candidates(color_filtered, candidates)
            if self.debug:
                print(f"[V3] Expanded to {len(expanded)} candidates by including nearby low-sat squares")
            if len(expanded) >= 9:
                candidates = expanded
            else:
                candidates = color_filtered
        elif len(color_filtered) >= 9:
            candidates = color_filtered

        # Step 3b: If we have more than 9 candidates, find the best cluster
        if len(candidates) > 9:
            candidates = self._find_best_grid_cluster(candidates, image.shape[:2])

        # Step 4: Filter candidates by neighbor relationships
        validated = self._validate_by_neighbors(candidates)

        if self.debug:
            print(f"[V3] Validated {len(validated)} facelets by neighbor analysis")

        # Step 3: Try to reconstruct 3x3 grid
        grid_facelets = self._reconstruct_grid(validated, image.shape[:2])

        if grid_facelets is not None and len(grid_facelets) == 9:
            # Successfully found all 9 facelets
            if self.debug:
                print("[V3] Successfully reconstructed 3x3 grid from facelets")
            return self._extract_region_from_grid(image, grid_facelets)

        # Step 4: Fallback - use detected facelets to estimate cube region
        if len(validated) >= 4:
            if self.debug:
                print(f"[V3] Using {len(validated)} facelets to estimate cube region")
            return self._estimate_region_from_facelets(image, validated)

        # Step 5: Ultimate fallback - saturation-based detection
        if self.debug:
            print("[V3] Falling back to saturation-based detection")
        return self._fallback_saturation_detection(image)

    def _find_square_contours(
        self,
        image: np.ndarray,
        relaxed: bool = False
    ) -> List[DetectedFacelet]:
        """
        Find all square-shaped contours in the image.

        Uses multiple edge detection approaches and combines results.
        """
        height, width = image.shape[:2]
        candidates = []

        # Expected facelet size range (cube takes up 30-80% of image, facelets are 1/3 of that)
        min_facelet = int(min(width, height) * 0.05)
        max_facelet = int(min(width, height) * 0.35)

        if relaxed:
            min_facelet = int(min(width, height) * 0.03)
            max_facelet = int(min(width, height) * 0.45)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try multiple preprocessing approaches
        preprocessed = []

        # Approach 1: Bilateral filter + Canny
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges1 = cv2.Canny(filtered, 30, 100)
        preprocessed.append(edges1)

        # Approach 2: Gaussian blur + adaptive threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        edges2 = cv2.Canny(thresh, 50, 150)
        preprocessed.append(edges2)

        # Approach 3: CLAHE enhancement + Canny
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges3 = cv2.Canny(enhanced, 40, 120)
        preprocessed.append(edges3)

        # Combine all edge images
        combined_edges = np.zeros_like(edges1)
        for edges in preprocessed:
            combined_edges = cv2.bitwise_or(combined_edges, edges)

        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(combined_edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            if w < min_facelet or h < min_facelet:
                continue
            if w > max_facelet or h > max_facelet:
                continue

            # Aspect ratio filter (should be approximately square)
            aspect = min(w, h) / max(w, h)
            if aspect < 0.6:  # Allow some perspective distortion
                continue

            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            # Should have 4 vertices (quadrilateral)
            if len(approx) < 4 or len(approx) > 6:
                # Try different epsilon values
                found_quad = False
                for eps in [0.02, 0.03, 0.05, 0.06]:
                    approx = cv2.approxPolyDP(contour, eps * peri, True)
                    if 4 <= len(approx) <= 6:
                        found_quad = True
                        break
                if not found_quad:
                    continue

            # Calculate area and filter
            area = cv2.contourArea(contour)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0

            # Real squares have high fill ratio
            if fill_ratio < 0.5:
                continue

            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2

            # Create candidate
            candidate = DetectedFacelet(
                contour=contour,
                center=(cx, cy),
                bounding_rect=(x, y, w, h),
                area=area,
                aspect_ratio=aspect,
                approx_poly=approx
            )

            # Check if this is a duplicate (overlapping with existing)
            is_duplicate = False
            for existing in candidates:
                ex, ey, ew, eh = existing.bounding_rect
                # Check overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                min_area = min(w * h, ew * eh)
                if overlap_area > 0.5 * min_area:
                    is_duplicate = True
                    # Keep the one with better aspect ratio
                    if candidate.aspect_ratio > existing.aspect_ratio:
                        candidates.remove(existing)
                        is_duplicate = False
                    break

            if not is_duplicate:
                candidates.append(candidate)

        return candidates

    def _analyze_candidate_colors(
        self,
        image: np.ndarray,
        candidates: List[DetectedFacelet]
    ) -> None:
        """
        Analyze the dominant color and saturation for each candidate.

        Sets the dominant_color field for each candidate.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for candidate in candidates:
            x, y, w, h = candidate.bounding_rect

            # Extract region with some margin inward to avoid borders
            margin = int(min(w, h) * 0.15)
            x1 = max(0, x + margin)
            y1 = max(0, y + margin)
            x2 = min(image.shape[1], x + w - margin)
            y2 = min(image.shape[0], y + h - margin)

            if x2 <= x1 or y2 <= y1:
                # Region too small
                candidate.dominant_color = np.array([0, 0, 128])  # Gray
                continue

            region_hsv = hsv[y1:y2, x1:x2]
            region_bgr = image[y1:y2, x1:x2]

            # Calculate average HSV values
            avg_h = np.mean(region_hsv[:, :, 0])
            avg_s = np.mean(region_hsv[:, :, 1])
            avg_v = np.mean(region_hsv[:, :, 2])

            # Store as (H, S, V)
            candidate.dominant_color = np.array([avg_h, avg_s, avg_v])

    def _filter_by_color(
        self,
        candidates: List[DetectedFacelet]
    ) -> List[DetectedFacelet]:
        """
        Filter candidates by color characteristics.

        Rubik's cube facelets are either:
        - Colored (red, orange, yellow, green, blue) with high saturation (S > 100)
        - White with low saturation (S < 40) and reasonable brightness (V > 160)

        This filters out gray keyboard keys (S < 20, V < 200) while keeping
        white cube facelets that may be in shadow.
        """
        filtered = []

        for candidate in candidates:
            if candidate.dominant_color is None:
                continue

            h, s, v = candidate.dominant_color

            # Accept if:
            # 1. High saturation (colored facelet) - S > 100
            # 2. White facelet (low saturation, good brightness) - S < 40 and V > 160
            # 3. Moderately saturated with good brightness

            is_colorful = s > 100  # Strong colors
            is_white = s < 40 and v > 160  # White (including shadowed)
            is_medium_color = s > 60 and v > 140  # Moderate saturation, reasonably lit

            # Reject if it looks like a keyboard key (very low sat, moderate brightness)
            is_keyboard_gray = s < 20 and v < 200

            if (is_colorful or is_white or is_medium_color) and not is_keyboard_gray:
                filtered.append(candidate)

        return filtered

    def _expand_with_nearby_candidates(
        self,
        colorful: List[DetectedFacelet],
        all_candidates: List[DetectedFacelet]
    ) -> List[DetectedFacelet]:
        """
        Expand colorful facelet set by including nearby low-saturation candidates.

        When we have colorful facelets but not enough for a full grid,
        find low-saturation candidates (potential white facelets) that are
        spatially close and have similar size to the colorful ones.
        """
        if not colorful:
            return colorful

        # Get statistics from colorful candidates
        colorful_centers = np.array([c.center for c in colorful])
        colorful_sizes = [max(c.bounding_rect[2], c.bounding_rect[3]) for c in colorful]
        median_size = np.median(colorful_sizes)
        size_tolerance = median_size * 0.4

        # Calculate bounding box of colorful candidates with margin
        min_x = np.min(colorful_centers[:, 0]) - median_size * 1.5
        max_x = np.max(colorful_centers[:, 0]) + median_size * 1.5
        min_y = np.min(colorful_centers[:, 1]) - median_size * 1.5
        max_y = np.max(colorful_centers[:, 1]) + median_size * 1.5

        expanded = list(colorful)
        colorful_set = set(id(c) for c in colorful)

        for candidate in all_candidates:
            if id(candidate) in colorful_set:
                continue

            cx, cy = candidate.center
            cw = max(candidate.bounding_rect[2], candidate.bounding_rect[3])

            # Check if within expanded bounding box
            if not (min_x <= cx <= max_x and min_y <= cy <= max_y):
                continue

            # Check if similar size
            if abs(cw - median_size) > size_tolerance:
                continue

            # Check if it has reasonable brightness (not too dark)
            if candidate.dominant_color is not None:
                h, s, v = candidate.dominant_color
                if v < 150:  # Too dark to be a white facelet
                    continue

            expanded.append(candidate)

        return expanded

    def _find_best_grid_cluster(
        self,
        candidates: List[DetectedFacelet],
        image_shape: Tuple[int, int]
    ) -> List[DetectedFacelet]:
        """
        Find the best cluster of 9 candidates that form a valid 3x3 grid.

        When there are multiple possible grid clusters (e.g., cube + keyboard),
        prefer the cluster with higher average saturation (more colorful).
        """
        if len(candidates) < 9:
            return candidates

        # Group candidates by spatial proximity
        # Calculate pairwise distances between candidates
        n = len(candidates)
        centers = np.array([c.center for c in candidates])

        # Get median size for distance threshold
        sizes = [max(c.bounding_rect[2], c.bounding_rect[3]) for c in candidates]
        median_size = np.median(sizes)

        # Find connected components using neighbor distance
        max_neighbor_dist = median_size * 2.5
        adjacency = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < max_neighbor_dist:
                    adjacency[i, j] = True
                    adjacency[j, i] = True

        # Find connected components using BFS
        visited = [False] * n
        clusters = []

        for start in range(n):
            if visited[start]:
                continue

            cluster = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                cluster.append(candidates[node])

                for neighbor in range(n):
                    if adjacency[node, neighbor] and not visited[neighbor]:
                        queue.append(neighbor)

            if len(cluster) >= 5:  # Minimum viable cluster
                clusters.append(cluster)

        if not clusters:
            return candidates

        # Score each cluster by average saturation
        best_cluster = None
        best_score = -1

        for cluster in clusters:
            # Calculate average saturation of cluster, excluding very low saturation items
            saturations = []
            high_sat_count = 0
            for c in cluster:
                if c.dominant_color is not None:
                    sat = c.dominant_color[1]
                    saturations.append(sat)
                    if sat > 100:  # Count highly saturated (definitely colorful)
                        high_sat_count += 1

            if saturations:
                avg_sat = np.mean(saturations)
                # Prefer clusters with more high-saturation candidates
                # This helps select the cube facelets over keyboard keys
                high_sat_ratio = high_sat_count / len(cluster)
                score = avg_sat * (0.5 + 0.5 * high_sat_ratio)

                if score > best_score:
                    best_score = score
                    best_cluster = cluster

        # After selecting best cluster, filter out very low saturation candidates
        # unless they're needed for a full grid (white facelets)
        if best_cluster is not None:
            high_sat = [c for c in best_cluster
                       if c.dominant_color is not None and c.dominant_color[1] > 50]
            low_sat = [c for c in best_cluster
                      if c.dominant_color is not None and c.dominant_color[1] <= 50]

            # If we have enough high-sat candidates, filter out suspicious low-sat ones
            if len(high_sat) >= 6:
                # Keep low-sat only if they're spatially consistent with high-sat
                if high_sat:
                    high_sat_y = [c.center[1] for c in high_sat]
                    min_y, max_y = min(high_sat_y), max(high_sat_y)
                    median_size = np.median([max(c.bounding_rect[2], c.bounding_rect[3])
                                            for c in high_sat])
                    # Keep low-sat candidates that fall within the Y range of high-sat
                    # Use smaller margin (0.5x) to avoid including keyboard keys
                    margin = median_size * 0.5
                    good_low_sat = [c for c in low_sat
                                   if min_y - margin <= c.center[1] <= max_y + margin]
                    best_cluster = high_sat + good_low_sat

            if self.debug:
                print(f"[V3] Selected cluster with {len(best_cluster)} candidates, avg saturation score: {best_score:.1f}")
            return best_cluster

        return candidates

    def _validate_by_neighbors(
        self,
        candidates: List[DetectedFacelet]
    ) -> List[DetectedFacelet]:
        """
        Validate facelet candidates by checking neighbor relationships.

        A real facelet should have neighboring facelets (unless it's at an edge).
        This filters out random squares that happen to be in the image.
        """
        if len(candidates) < 4:
            return candidates  # Not enough to validate

        # Calculate median facelet size for distance threshold
        sizes = [max(c.bounding_rect[2], c.bounding_rect[3]) for c in candidates]
        median_size = np.median(sizes)

        # Neighbor distance: facelets should be roughly 1 facelet-width apart
        # Allow some tolerance for gaps (black borders) and perspective
        min_neighbor_dist = median_size * 0.5
        max_neighbor_dist = median_size * 2.0

        # Count neighbors for each candidate
        for candidate in candidates:
            cx, cy = candidate.center
            neighbor_count = 0

            for other in candidates:
                if other is candidate:
                    continue

                ox, oy = other.center
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)

                if min_neighbor_dist < dist < max_neighbor_dist:
                    # Check if neighbor is roughly aligned (horizontal or vertical)
                    dx = abs(cx - ox)
                    dy = abs(cy - oy)

                    # Either horizontal or vertical alignment
                    if dx < median_size * 0.5 or dy < median_size * 0.5:
                        neighbor_count += 1

            candidate.neighbor_count = neighbor_count

        # Filter: keep candidates with at least 1 neighbor
        # (edge and corner facelets have 2-3 neighbors, center has 4)
        validated = [c for c in candidates if c.neighbor_count >= 1]

        # If too few validated, try with size consistency filter
        if len(validated) < 5:
            # Try keeping all candidates with similar sizes
            size_tolerance = median_size * 0.3
            similar_size = [
                c for c in candidates
                if abs(max(c.bounding_rect[2], c.bounding_rect[3]) - median_size) < size_tolerance
            ]
            if len(similar_size) > len(validated):
                validated = similar_size

        return validated

    def _reconstruct_grid(
        self,
        facelets: List[DetectedFacelet],
        image_shape: Tuple[int, int]
    ) -> Optional[List[DetectedFacelet]]:
        """
        Try to reconstruct a 3x3 grid from detected facelets.

        Assigns grid positions (row, col) to each facelet based on
        their spatial relationships.
        """
        if len(facelets) < 5:
            return None

        # Get centers
        centers = np.array([f.center for f in facelets])

        # Find approximate grid spacing
        sizes = [max(f.bounding_rect[2], f.bounding_rect[3]) for f in facelets]
        median_size = np.median(sizes)

        # Cluster X coordinates to find columns
        x_coords = sorted(centers[:, 0])
        x_clusters = self._cluster_1d(x_coords, median_size * 0.5)

        # Cluster Y coordinates to find rows
        y_coords = sorted(centers[:, 1])
        y_clusters = self._cluster_1d(y_coords, median_size * 0.5)

        if self.debug:
            print(f"[V3] Found {len(x_clusters)} column clusters, {len(y_clusters)} row clusters")

        # We need exactly 3 rows and 3 columns
        if len(x_clusters) != 3 or len(y_clusters) != 3:
            # Try to infer missing clusters
            x_clusters = self._infer_missing_clusters(x_clusters, 3, median_size)
            y_clusters = self._infer_missing_clusters(y_clusters, 3, median_size)

        if len(x_clusters) != 3 or len(y_clusters) != 3:
            return None

        # Sort clusters
        x_clusters = sorted(x_clusters)
        y_clusters = sorted(y_clusters)

        # Assign grid positions to each facelet
        grid = [[None for _ in range(3)] for _ in range(3)]
        assigned_facelets = []

        for facelet in facelets:
            cx, cy = facelet.center

            # Find closest column
            col = min(range(3), key=lambda i: abs(cx - x_clusters[i]))
            # Find closest row
            row = min(range(3), key=lambda i: abs(cy - y_clusters[i]))

            # Check if this position is reasonable
            dist_to_col = abs(cx - x_clusters[col])
            dist_to_row = abs(cy - y_clusters[row])

            if dist_to_col < median_size * 0.7 and dist_to_row < median_size * 0.7:
                if grid[row][col] is None:
                    grid[row][col] = facelet
                    facelet.grid_position = (row, col)
                    assigned_facelets.append(facelet)
                else:
                    # Position already taken - keep the one closer to expected position
                    existing = grid[row][col]
                    ex, ey = existing.center
                    existing_dist = abs(ex - x_clusters[col]) + abs(ey - y_clusters[row])
                    new_dist = dist_to_col + dist_to_row
                    if new_dist < existing_dist:
                        assigned_facelets.remove(existing)
                        grid[row][col] = facelet
                        facelet.grid_position = (row, col)
                        assigned_facelets.append(facelet)

        # Count how many positions are filled
        filled = sum(1 for row in grid for cell in row if cell is not None)

        if self.debug:
            print(f"[V3] Grid reconstruction: {filled}/9 positions filled")

        if filled >= 5:  # At least 5 facelets needed to estimate the rest
            return assigned_facelets

        return None

    def _cluster_1d(self, values: List[float], threshold: float) -> List[float]:
        """
        Cluster 1D values that are close together.

        Returns cluster centers.
        """
        if not values:
            return []

        clusters = []
        current_cluster = [values[0]]

        for val in values[1:]:
            if val - current_cluster[-1] < threshold:
                current_cluster.append(val)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [val]

        if current_cluster:
            clusters.append(np.mean(current_cluster))

        return clusters

    def _infer_missing_clusters(
        self,
        clusters: List[float],
        target_count: int,
        expected_spacing: float
    ) -> List[float]:
        """
        Infer missing cluster positions based on expected spacing.
        """
        if len(clusters) == target_count:
            return clusters

        if len(clusters) < 2:
            return clusters

        clusters = sorted(clusters)

        # Calculate actual spacing between existing clusters
        spacings = [clusters[i+1] - clusters[i] for i in range(len(clusters)-1)]
        avg_spacing = np.mean(spacings)

        # Try to add missing clusters
        while len(clusters) < target_count:
            # Check if there's a gap approximately 2x the expected spacing
            added = False
            for i in range(len(clusters) - 1):
                gap = clusters[i+1] - clusters[i]
                if gap > avg_spacing * 1.5:
                    # Insert a cluster in the middle
                    new_pos = (clusters[i] + clusters[i+1]) / 2
                    clusters.insert(i + 1, new_pos)
                    added = True
                    break

            if not added:
                # Try adding at the ends
                if len(clusters) < target_count:
                    # Add at beginning
                    clusters.insert(0, clusters[0] - avg_spacing)
                if len(clusters) < target_count:
                    # Add at end
                    clusters.append(clusters[-1] + avg_spacing)

            if len(clusters) >= target_count:
                break

        return clusters[:target_count]

    def _extract_region_from_grid(
        self,
        image: np.ndarray,
        grid_facelets: List[DetectedFacelet]
    ) -> np.ndarray:
        """
        Extract the cube face region based on detected grid facelets.

        Calculates the bounding box of all facelets plus margin for the
        black borders.
        """
        # Find bounding box of all facelets
        all_x = []
        all_y = []

        for facelet in grid_facelets:
            x, y, w, h = facelet.bounding_rect
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])

        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)

        # Add margin for outer black border
        median_size = np.median([max(f.bounding_rect[2], f.bounding_rect[3])
                                  for f in grid_facelets])
        margin = int(median_size * 0.1)

        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)

        # Make it square
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        x_min = max(0, cx - size // 2)
        y_min = max(0, cy - size // 2)
        x_max = min(image.shape[1], x_min + size)
        y_max = min(image.shape[0], y_min + size)

        # Ensure square
        actual_width = x_max - x_min
        actual_height = y_max - y_min
        final_size = min(actual_width, actual_height)

        region = image[y_min:y_min+final_size, x_min:x_min+final_size]

        return region

    def _estimate_region_from_facelets(
        self,
        image: np.ndarray,
        facelets: List[DetectedFacelet]
    ) -> np.ndarray:
        """
        Estimate the cube region when we have some but not all facelets detected.

        Uses the detected facelets to estimate the full cube extent.
        """
        # Get median facelet size
        sizes = [max(f.bounding_rect[2], f.bounding_rect[3]) for f in facelets]
        median_size = np.median(sizes)

        # Estimate full cube size (3x3 facelets + borders)
        estimated_cube_size = int(median_size * 3.3)  # 3 facelets + gaps

        # Find center of detected facelets
        centers = np.array([f.center for f in facelets])
        center_x = np.mean(centers[:, 0])
        center_y = np.mean(centers[:, 1])

        # Extract square region centered on facelets
        half = estimated_cube_size // 2
        x_min = max(0, int(center_x - half))
        y_min = max(0, int(center_y - half))
        x_max = min(image.shape[1], x_min + estimated_cube_size)
        y_max = min(image.shape[0], y_min + estimated_cube_size)

        # Ensure square
        actual_size = min(x_max - x_min, y_max - y_min)
        region = image[y_min:y_min+actual_size, x_min:x_min+actual_size]

        return region

    def _fallback_saturation_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback detection using color saturation.

        When contour-based detection fails, fall back to finding
        the most colorful region in the image.
        """
        height, width = image.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Find region with highest saturation
        best_region = None
        best_score = 0

        for size_pct in [0.6, 0.5, 0.45, 0.7]:
            size = int(min(width, height) * size_pct)
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

    def _extract_facelets(self, face_region: np.ndarray) -> List[np.ndarray]:
        """
        Extract 9 facelets from the face region.

        First tries to detect grid lines, falls back to uniform splitting.
        """
        height, width = face_region.shape[:2]

        # Try to detect grid lines using dark pixels (black borders)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Find dark pixels (black plastic borders)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Project onto axes to find grid lines
        h_projection = np.sum(dark_mask, axis=1).astype(float)
        v_projection = np.sum(dark_mask, axis=0).astype(float)

        # Find grid lines
        h_lines = self._find_grid_lines(h_projection, height)
        v_lines = self._find_grid_lines(v_projection, width)

        if h_lines is not None and v_lines is not None:
            # Use detected grid lines
            return self._extract_by_grid_lines(face_region, h_lines, v_lines)
        else:
            # Fall back to uniform splitting
            return self._extract_uniform(face_region)

    def _find_grid_lines(
        self,
        projection: np.ndarray,
        dimension: int
    ) -> Optional[List[float]]:
        """
        Find 4 grid lines (boundaries) from a projection.
        """
        # Smooth projection
        kernel_size = max(3, dimension // 30)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(projection, kernel, mode='same')

        # Normalize
        if smoothed.max() == 0:
            return None
        smoothed = smoothed / smoothed.max()

        # Find peaks
        threshold = 0.3
        margin = dimension // 8

        peaks = []
        in_peak = False
        peak_start = 0

        for i in range(margin, dimension - margin):
            if smoothed[i] > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif smoothed[i] <= threshold and in_peak:
                in_peak = False
                peaks.append((peak_start + i) // 2)

        # Need at least 2 internal lines
        if len(peaks) < 2:
            return None

        # Add boundaries
        all_lines = [0] + peaks + [dimension]

        # Find 4 evenly spaced lines
        return self._find_evenly_spaced(all_lines, dimension)

    def _find_evenly_spaced(
        self,
        positions: List[float],
        max_dim: int
    ) -> Optional[List[float]]:
        """
        Find 4 evenly-spaced positions from candidates.
        """
        if len(positions) < 4:
            return None

        positions = sorted(positions)

        best_lines = None
        best_score = float('inf')

        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for m in range(k + 1, n):
                        lines = [positions[i], positions[j], positions[k], positions[m]]

                        spacings = [lines[1] - lines[0],
                                   lines[2] - lines[1],
                                   lines[3] - lines[2]]

                        avg_spacing = np.mean(spacings)

                        if avg_spacing < max_dim * 0.15 or avg_spacing > max_dim * 0.5:
                            continue

                        variance = np.var(spacings)
                        score = variance / (avg_spacing ** 2) if avg_spacing > 0 else float('inf')

                        if score < best_score:
                            best_score = score
                            best_lines = lines

        if best_lines is not None and best_score < 0.15:
            return best_lines

        return None

    def _extract_by_grid_lines(
        self,
        face_region: np.ndarray,
        h_lines: List[float],
        v_lines: List[float]
    ) -> List[np.ndarray]:
        """
        Extract facelets using detected grid lines.
        """
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

    def _extract_uniform(self, face_region: np.ndarray) -> List[np.ndarray]:
        """
        Extract facelets using uniform 3x3 splitting.
        """
        height, width = face_region.shape[:2]
        facelet_h = height // 3
        facelet_w = width // 3

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

    def _extract_bbox_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract region from bounding box, handling rotation if present.
        """
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        img_h, img_w = image.shape[:2]

        if abs(bbox.rotation) > 0.5:
            # Handle rotated bbox with perspective transform
            cx, cy = x + w / 2, y + h / 2
            angle_rad = np.radians(bbox.rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            hw, hh = w / 2, h / 2
            corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_corners = corners @ rotation_matrix.T + np.array([cx, cy])

            src_points = rotated_corners.astype(np.float32)
            dst_points = np.array([
                [0, 0], [w, 0], [w, h], [0, h]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_points, dst_points)
            return cv2.warpPerspective(image, M, (int(w), int(h)))

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

    # Convenience methods for compatibility with v1/v2 interface

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


# Functional interface for compatibility
def segment_cube_face_v3(
    image: np.ndarray,
    bbox: Optional[BoundingBox] = None,
    output_size: int = 64
) -> np.ndarray:
    """
    Functional interface for v3 facelet segmentation.

    Args:
        image: Input image containing a Rubik's cube face (BGR format)
        bbox: Optional bounding box for the face region
        output_size: Size of output facelet images (default 64)

    Returns:
        numpy array of shape (3, 3, output_size, output_size, 3)
    """
    segmenter = FaceletSegmenterV3(output_size=output_size)
    return segmenter.segment(image, bbox)


# Alias for main class name compatibility
FaceletSegmenter = FaceletSegmenterV3
