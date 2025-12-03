"""
Rubik's Cube Face Segmentation - Version 4

Uses the standard OpenCV approach for square detection:
1. Canny edge detection
2. Contour finding
3. Filter contours for square-like shapes
4. Identify 9 squares arranged in a 3x3 grid
5. Extract each facelet

This approach directly detects the 9 facelets rather than detecting the
cube boundary and subdividing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations


@dataclass
class Square:
    """Represents a detected square region."""
    center: Tuple[int, int]  # (x, y) center point
    contour: np.ndarray      # Original contour points
    area: float              # Contour area
    approx: np.ndarray       # Approximated polygon (4 points)


@dataclass
class BoundingBox:
    """Represents a bounding box for the cube face region."""
    x: int
    y: int
    width: int
    height: int
    rotation: float = 0.0


class FaceletSegmenterV4:
    """
    Version 4 segmenter using standard OpenCV square detection.

    Approach:
    1. Apply bilateral filter to reduce noise while preserving edges
    2. Use Canny edge detection with automatic thresholding
    3. Find contours and filter for square-like shapes
    4. Group squares into a 3x3 grid pattern
    5. Extract each facelet from the grid

    This is more robust than boundary detection because it directly
    finds the colored squares rather than relying on the outer boundary.
    """

    def __init__(self, output_size: int = 64, debug: bool = False):
        """
        Initialize the segmenter.

        Args:
            output_size: Size of output facelet images (default 64x64)
            debug: If True, print debug information
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
            bbox: Optional bounding box (ignored in v4, kept for API compatibility)

        Returns:
            numpy array of shape (3, 3, 64, 64, 3)
        """
        # Strategy 1: Try hierarchical contour detection (finds cube frame with facelet holes)
        facelets = self._detect_by_frame_holes(image)

        if facelets is not None:
            if self.debug:
                print(f"[V4] Found cube using frame hole detection")
        else:
            # Strategy 2: Try to detect 9 squares in a grid pattern
            squares = self._detect_squares(image)
            grid = self._arrange_into_grid(squares, image.shape)

            if grid is not None:
                if self.debug:
                    print(f"[V4] Found 9-square grid")
                facelets = self._extract_facelets_from_grid(image, grid)
            else:
                # Strategy 3: Use grid line detection
                if self.debug:
                    print(f"[V4] Trying grid line detection")
                facelets = self._detect_by_grid_lines(image)

                if facelets is None:
                    if self.debug:
                        print(f"[V4] Grid line detection failed, using fallback")
                    # Strategy 4: Fallback to region detection
                    facelets = self._fallback_detection(image)

        # Resize each facelet to output size
        resized = []
        for facelet in facelets:
            resized.append(
                cv2.resize(facelet, (self.output_size, self.output_size),
                          interpolation=cv2.INTER_AREA)
            )

        # Reshape to 3x3 grid
        facelets_array = np.array(resized, dtype=np.uint8)
        return facelets_array.reshape(3, 3, self.output_size, self.output_size, 3)

    def _detect_by_frame_holes(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Detect cube by finding the dark frame with facelet holes.

        Uses hierarchical contour detection to find a dark region (cube frame)
        that contains multiple child contours (the facelet holes). Then uses
        the centers of these holes to establish the 3x3 grid positions.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find dark areas (cube frame is dark)
        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Morphological closing to connect the frame
        kernel = np.ones((5, 5), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(dark_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            return None

        # Find the cube frame (has multiple children = facelets)
        cube_contour = None
        cube_children = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 3000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if aspect < 0.5:
                continue

            # Count and collect children (facelet holes)
            children = []
            child_idx = hierarchy[0][i][2]  # First child index
            while child_idx != -1:
                child_cnt = contours[child_idx]
                child_area = cv2.contourArea(child_cnt)
                if child_area > 200:  # Filter tiny noise
                    cx, cy, cw, ch = cv2.boundingRect(child_cnt)
                    children.append({
                        'contour': child_cnt,
                        'bbox': (cx, cy, cw, ch),
                        'center': (cx + cw // 2, cy + ch // 2),
                        'area': child_area
                    })
                child_idx = hierarchy[0][child_idx][0]  # Next sibling

            # Need at least 5 visible facelets to establish the grid
            if len(children) >= 5:
                if cube_contour is None or len(children) > len(cube_children):
                    cube_contour = cnt
                    cube_children = children

        if cube_contour is None or len(cube_children) < 5:
            return None

        # Get cube bounding box
        cx, cy, cw, ch = cv2.boundingRect(cube_contour)

        # Use detected facelets to establish the grid
        centers = np.array([c['center'] for c in cube_children])

        # Cluster x and y coordinates to find grid positions
        x_clusters = self._cluster_1d(centers[:, 0], cw)
        y_clusters = self._cluster_1d(centers[:, 1], ch)

        # Need at least 2 positions in each dimension
        if len(x_clusters) < 2 or len(y_clusters) < 2:
            return None

        # Estimate cell spacing
        x_spacing = np.mean(np.diff(sorted(x_clusters))) if len(x_clusters) >= 2 else cw / 3
        y_spacing = np.mean(np.diff(sorted(y_clusters))) if len(y_clusters) >= 2 else ch / 3

        # Fill in missing clusters to get exactly 3
        x_clusters = self._fill_clusters(x_clusters, x_spacing, cx, cx + cw)
        y_clusters = self._fill_clusters(y_clusters, y_spacing, cy, cy + ch)

        if len(x_clusters) != 3 or len(y_clusters) != 3:
            return None

        # Calculate facelet size from spacing
        facelet_w = int(x_spacing * 0.9)
        facelet_h = int(y_spacing * 0.9)

        # Extract facelets centered on grid positions
        facelets = []
        for yc in y_clusters:
            for xc in x_clusters:
                fx = int(xc - facelet_w // 2)
                fy = int(yc - facelet_h // 2)

                # Clip to image bounds
                fx = max(0, min(fx, width - facelet_w))
                fy = max(0, min(fy, height - facelet_h))

                facelet = image[fy:fy + facelet_h, fx:fx + facelet_w]
                if facelet.size > 0:
                    facelets.append(facelet)
                else:
                    return None

        if len(facelets) != 9:
            return None

        return facelets

    def _cluster_1d(self, values: np.ndarray, dimension: float) -> List[float]:
        """Cluster 1D values into groups."""
        if len(values) == 0:
            return []

        values = sorted(values)
        tolerance = dimension * 0.15  # Values within 15% are same cluster

        clusters = []
        current = [values[0]]

        for v in values[1:]:
            if v - current[-1] < tolerance:
                current.append(v)
            else:
                clusters.append(np.mean(current))
                current = [v]
        clusters.append(np.mean(current))

        return sorted(clusters)

    def _fill_clusters(self, clusters: List[float], spacing: float,
                       min_bound: float, max_bound: float) -> List[float]:
        """Fill in missing clusters to get exactly 3 positions."""
        clusters = sorted(clusters)

        while len(clusters) < 3:
            if len(clusters) == 1:
                # Add positions on both sides
                clusters = [clusters[0] - spacing, clusters[0], clusters[0] + spacing]
            elif len(clusters) == 2:
                gap = clusters[1] - clusters[0]
                if gap > spacing * 1.5:
                    # Missing middle
                    clusters = [clusters[0], (clusters[0] + clusters[1]) / 2, clusters[1]]
                elif clusters[0] - min_bound > spacing * 0.5:
                    # Missing left
                    clusters = [clusters[0] - spacing] + clusters
                else:
                    # Missing right
                    clusters = clusters + [clusters[1] + spacing]
            else:
                break

        return sorted(clusters[:3])

    def _detect_by_grid_lines(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Detect cube face by finding the grid pattern of black lines.

        Uses Hough line detection to find horizontal and vertical lines,
        then identifies the 3x3 grid structure.
        """
        height, width = image.shape[:2]

        # First, find the cube region using color
        cube_region = self._find_cube_region(image)
        if cube_region is None:
            return None

        region_img, (rx, ry, rw, rh) = cube_region
        rh_img, rw_img = region_img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=rw_img//4, maxLineGap=rw_img//10)

        if lines is None:
            return None

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Horizontal lines (angle near 0 or 180)
            if abs(angle) < 15 or abs(angle) > 165:
                h_lines.append((y1 + y2) / 2)  # Average y position
            # Vertical lines (angle near 90 or -90)
            elif abs(abs(angle) - 90) < 15:
                v_lines.append((x1 + x2) / 2)  # Average x position

        # Cluster line positions to find grid lines
        h_positions = self._cluster_positions(h_lines, rh_img)
        v_positions = self._cluster_positions(v_lines, rw_img)

        # We need at least 2 horizontal and 2 vertical internal lines
        if len(h_positions) < 2 or len(v_positions) < 2:
            return None

        # Add edge positions and sort
        h_positions = sorted([0] + list(h_positions) + [rh_img])
        v_positions = sorted([0] + list(v_positions) + [rw_img])

        # Check if we have a valid 3x3 grid (4 horizontal, 4 vertical positions)
        if len(h_positions) < 4 or len(v_positions) < 4:
            return None

        # Use the most evenly spaced 4 positions
        h_grid = self._select_even_positions(h_positions, 4)
        v_grid = self._select_even_positions(v_positions, 4)

        if h_grid is None or v_grid is None:
            return None

        # Extract facelets from grid
        facelets = []
        for row in range(3):
            for col in range(3):
                y1 = int(h_grid[row])
                y2 = int(h_grid[row + 1])
                x1 = int(v_grid[col])
                x2 = int(v_grid[col + 1])

                # Add margin to avoid borders
                margin_h = int((y2 - y1) * 0.1)
                margin_w = int((x2 - x1) * 0.1)

                y1 = min(y1 + margin_h, y2 - 1)
                y2 = max(y2 - margin_h, y1 + 1)
                x1 = min(x1 + margin_w, x2 - 1)
                x2 = max(x2 - margin_w, x1 + 1)

                facelet = region_img[y1:y2, x1:x2]
                if facelet.size > 0:
                    facelets.append(facelet)
                else:
                    # Return None if any facelet is invalid
                    return None

        if len(facelets) != 9:
            return None

        return facelets

    def _find_cube_region(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Find the cube region in the image using color saturation and edges."""
        height, width = image.shape[:2]

        # Use combined saturation + edge approach (similar to V1/V2)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Lower saturation threshold
        _, sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 60, 180)

        # Fill edges
        kernel = np.ones((7, 7), np.uint8)
        edges_filled = cv2.dilate(edges, kernel, iterations=4)
        edges_filled = cv2.morphologyEx(edges_filled, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Combine saturation and edges
        combined_mask = cv2.addWeighted(sat_mask, 0.7, edges_filled, 0.3, 0)
        _, combined_mask = cv2.threshold(combined_mask, 100, 255, cv2.THRESH_BINARY)

        # Cleanup
        kernel_large = np.ones((11, 11), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find best cube region
        min_size = min(width, height) * 0.15
        max_size = min(width, height) * 0.85  # Don't allow regions that cover entire image
        best = None
        best_score = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_size or h < min_size:
                continue
            if w > max_size and h > max_size:
                continue  # Skip if covers entire image

            squareness = min(w, h) / max(w, h)
            roi_sat = saturation[y:y+h, x:x+w]
            avg_saturation = roi_sat.mean()

            if avg_saturation < 40:
                continue
            if squareness < 0.7:
                continue

            # Score: prefer square, colorful, and reasonably sized regions
            score = squareness * avg_saturation * (1 - abs(w*h - (width*height/4)) / (width*height))

            if score > best_score:
                best_score = score
                # Make square
                size = max(w, h)
                size = min(size, min(width, height))  # Don't exceed image
                cx, cy = x + w // 2, y + h // 2
                nx = max(0, min(cx - size // 2, width - size))
                ny = max(0, min(cy - size // 2, height - size))
                # Ensure we don't go out of bounds
                size = min(size, width - nx, height - ny)
                if size > min_size:
                    best = (image[ny:ny+size, nx:nx+size], (nx, ny, size, size))

        return best

    def _cluster_positions(self, positions: List[float], dimension: int) -> List[float]:
        """Cluster nearby positions and return cluster centers."""
        if not positions:
            return []

        sorted_pos = sorted(positions)
        clusters = []
        current_cluster = [sorted_pos[0]]

        threshold = dimension * 0.1  # Positions within 10% are same cluster

        for pos in sorted_pos[1:]:
            if pos - current_cluster[-1] < threshold:
                current_cluster.append(pos)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [pos]

        clusters.append(np.mean(current_cluster))
        return clusters

    def _select_even_positions(self, positions: List[float], count: int) -> Optional[List[float]]:
        """Select 'count' positions that are most evenly spaced."""
        if len(positions) < count:
            return None

        if len(positions) == count:
            return positions

        # Try all combinations of 'count' positions
        best_positions = None
        best_variance = float('inf')

        for combo in combinations(range(len(positions)), count):
            selected = [positions[i] for i in combo]
            # Must include first and last
            if selected[0] != positions[0] or selected[-1] != positions[-1]:
                continue

            # Calculate spacing variance
            spacings = [selected[i+1] - selected[i] for i in range(len(selected)-1)]
            variance = np.var(spacings)

            if variance < best_variance:
                best_variance = variance
                best_positions = selected

        return best_positions

    def _detect_squares(self, image: np.ndarray) -> List[Square]:
        """
        Detect square-like contours in the image.

        Uses multiple preprocessing strategies to find squares robustly.
        """
        height, width = image.shape[:2]
        all_squares = []

        # Strategy 1: Canny on grayscale with bilateral filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Auto-threshold Canny using median
        median = np.median(blurred)
        lower = int(max(0, 0.67 * median))
        upper = int(min(255, 1.33 * median))
        edges1 = cv2.Canny(blurred, lower, upper)

        # Strategy 2: Canny with fixed thresholds (for high contrast images)
        edges2 = cv2.Canny(blurred, 50, 150)

        # Strategy 3: Adaptive threshold + Canny (for varying lighting)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        edges3 = cv2.Canny(adaptive, 50, 150)

        # Strategy 4: Color-based edge detection (detects colored squares better)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)
        edges4 = cv2.Canny(sat_mask, 50, 150)

        # Combine edges from all strategies
        for edges in [edges1, edges2, edges3, edges4]:
            # Dilate to connect broken edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                squares_found = self._check_square(contour, width, height)
                all_squares.extend(squares_found)

        # Remove duplicates (squares with very similar centers)
        unique_squares = self._remove_duplicate_squares(all_squares)

        if self.debug:
            print(f"[V4] Found {len(unique_squares)} unique squares")

        return unique_squares

    def _check_square(self, contour: np.ndarray, img_width: int, img_height: int) -> List[Square]:
        """
        Check if a contour is a valid square and return Square object if so.
        """
        squares = []
        area = cv2.contourArea(contour)

        # Filter by area - squares should be reasonable size
        # Each facelet is roughly 1/9 of the cube, cube is 20-70% of image
        min_area = (min(img_width, img_height) * 0.05) ** 2
        max_area = (min(img_width, img_height) * 0.35) ** 2

        if area < min_area or area > max_area:
            return squares

        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Must have 4 vertices
        if len(approx) != 4:
            return squares

        # Check if it's convex
        if not cv2.isContourConvex(approx):
            return squares

        # Check aspect ratio (should be close to 1 for a square)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        if aspect_ratio < 0.7:
            return squares

        # Check that angles are close to 90 degrees
        if not self._has_right_angles(approx):
            return squares

        # Calculate center
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        squares.append(Square(
            center=(cx, cy),
            contour=contour,
            area=area,
            approx=approx.reshape(4, 2)
        ))

        return squares

    def _has_right_angles(self, approx: np.ndarray) -> bool:
        """Check if the quadrilateral has approximately right angles."""
        pts = approx.reshape(4, 2)

        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]

            # Vectors from p2 to p1 and p2 to p3
            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate angle using dot product
            dot = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)

            if mag1 == 0 or mag2 == 0:
                return False

            cos_angle = dot / (mag1 * mag2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))

            # Allow 15 degree tolerance from 90
            if abs(angle - 90) > 15:
                return False

        return True

    def _remove_duplicate_squares(self, squares: List[Square]) -> List[Square]:
        """Remove squares with very similar centers."""
        if not squares:
            return []

        # Sort by area (larger first)
        squares = sorted(squares, key=lambda s: s.area, reverse=True)

        unique = []
        for sq in squares:
            is_duplicate = False
            for existing in unique:
                dist = np.sqrt((sq.center[0] - existing.center[0])**2 +
                              (sq.center[1] - existing.center[1])**2)
                # If centers are within 10% of the square size, consider duplicate
                size = np.sqrt(sq.area)
                if dist < size * 0.3:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(sq)

        return unique

    def _arrange_into_grid(self, squares: List[Square],
                           img_shape: Tuple[int, ...]) -> Optional[List[List[Square]]]:
        """
        Try to arrange detected squares into a 3x3 grid.

        Returns a 3x3 list of Squares if successful, None otherwise.
        """
        if len(squares) < 9:
            if self.debug:
                print(f"[V4] Only {len(squares)} squares found, need 9")
            return None

        # Find groups of 9 squares that form a grid
        # Strategy: find squares of similar size, then check grid arrangement

        # Group squares by similar area (within 50%)
        area_groups = self._group_by_area(squares)

        for group in area_groups:
            if len(group) < 9:
                continue

            # Try to find 9 squares forming a grid
            grid = self._find_grid_in_group(group)
            if grid is not None:
                return grid

        return None

    def _group_by_area(self, squares: List[Square]) -> List[List[Square]]:
        """Group squares by similar area."""
        if not squares:
            return []

        # Sort by area
        sorted_squares = sorted(squares, key=lambda s: s.area)

        groups = []
        current_group = [sorted_squares[0]]

        for sq in sorted_squares[1:]:
            # Check if area is within 50% of group average
            avg_area = np.mean([s.area for s in current_group])
            if sq.area <= avg_area * 2.0 and sq.area >= avg_area * 0.5:
                current_group.append(sq)
            else:
                if len(current_group) >= 9:
                    groups.append(current_group)
                current_group = [sq]

        if len(current_group) >= 9:
            groups.append(current_group)

        return groups

    def _find_grid_in_group(self, squares: List[Square]) -> Optional[List[List[Square]]]:
        """
        Find 9 squares arranged in a 3x3 grid pattern.
        """
        if len(squares) < 9:
            return None

        # Get centers
        centers = np.array([sq.center for sq in squares])

        # Try to find grid structure using clustering
        # For a 3x3 grid, we expect 3 clusters in x and 3 clusters in y

        # Sort by x-coordinate and find 3 column groups
        x_sorted_idx = np.argsort(centers[:, 0])

        # Sort by y-coordinate and find 3 row groups
        y_sorted_idx = np.argsort(centers[:, 1])

        # Try combinations of 9 squares
        best_grid = None
        best_score = float('inf')

        # Use the 9 squares with most regular spacing
        for start_idx in range(min(len(squares) - 8, 5)):
            candidate_squares = squares[start_idx:start_idx + 9]
            if len(candidate_squares) < 9:
                continue

            grid, score = self._try_form_grid(candidate_squares)
            if grid is not None and score < best_score:
                best_grid = grid
                best_score = score

        # Also try using the squares with most similar areas
        areas = [sq.area for sq in squares]
        median_area = np.median(areas)
        area_diff = [abs(a - median_area) for a in areas]
        best_area_idx = np.argsort(area_diff)[:9]

        candidate_squares = [squares[i] for i in best_area_idx]
        grid, score = self._try_form_grid(candidate_squares)
        if grid is not None and score < best_score:
            best_grid = grid
            best_score = score

        return best_grid

    def _try_form_grid(self, squares: List[Square]) -> Tuple[Optional[List[List[Square]]], float]:
        """
        Try to arrange exactly 9 squares into a 3x3 grid.

        Returns (grid, regularity_score) or (None, inf) if not a valid grid.
        """
        if len(squares) != 9:
            return None, float('inf')

        centers = np.array([sq.center for sq in squares])

        # Sort by y first (rows), then by x (columns)
        # This gives us row-major ordering

        # Find row assignments using y-coordinate clustering
        y_coords = centers[:, 1]
        y_sorted_idx = np.argsort(y_coords)

        # Assign to 3 rows (first 3, middle 3, last 3 when sorted by y)
        rows = [[], [], []]
        row_y_means = []

        # Simple approach: divide sorted y-values into 3 groups
        for i, idx in enumerate(y_sorted_idx):
            row_idx = min(i // 3, 2)
            rows[row_idx].append((idx, squares[idx]))

        # Check each row has exactly 3 squares
        for row in rows:
            if len(row) != 3:
                return None, float('inf')

        # Sort each row by x-coordinate
        grid = []
        for row in rows:
            sorted_row = sorted(row, key=lambda item: item[1].center[0])
            grid.append([item[1] for item in sorted_row])

        # Calculate regularity score
        # A good grid has:
        # 1. Equal spacing between columns
        # 2. Equal spacing between rows
        # 3. Aligned rows and columns

        score = self._calculate_grid_regularity(grid)

        # Threshold for valid grid
        avg_size = np.mean([sq.area for sq in squares])
        spacing_threshold = np.sqrt(avg_size) * 0.5

        if score > spacing_threshold:
            return None, float('inf')

        return grid, score

    def _calculate_grid_regularity(self, grid: List[List[Square]]) -> float:
        """
        Calculate how regular/uniform a 3x3 grid is.

        Lower score = more regular grid.
        """
        # Check column alignment (x-coordinates in each column should be similar)
        col_x_variance = 0
        for col in range(3):
            x_coords = [grid[row][col].center[0] for row in range(3)]
            col_x_variance += np.var(x_coords)

        # Check row alignment (y-coordinates in each row should be similar)
        row_y_variance = 0
        for row in range(3):
            y_coords = [grid[row][col].center[1] for col in range(3)]
            row_y_variance += np.var(y_coords)

        # Check spacing uniformity
        h_spacings = []
        v_spacings = []

        for row in range(3):
            for col in range(2):
                h_spacings.append(grid[row][col + 1].center[0] - grid[row][col].center[0])

        for col in range(3):
            for row in range(2):
                v_spacings.append(grid[row + 1][col].center[1] - grid[row][col].center[1])

        h_spacing_variance = np.var(h_spacings) if h_spacings else 0
        v_spacing_variance = np.var(v_spacings) if v_spacings else 0

        # Combined score
        score = np.sqrt(col_x_variance + row_y_variance + h_spacing_variance + v_spacing_variance)

        return score

    def _extract_facelets_from_grid(self, image: np.ndarray,
                                     grid: List[List[Square]]) -> List[np.ndarray]:
        """
        Extract facelet images from the detected grid.
        """
        facelets = []

        for row in range(3):
            for col in range(3):
                square = grid[row][col]
                facelet = self._extract_square_region(image, square)
                facelets.append(facelet)

        return facelets

    def _extract_square_region(self, image: np.ndarray, square: Square) -> np.ndarray:
        """
        Extract the region inside a square, with perspective correction.
        """
        pts = square.approx.astype(np.float32)

        # Order points: top-left, top-right, bottom-right, bottom-left
        ordered = self._order_points(pts)

        # Calculate output size based on square dimensions
        width = int(max(
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[3])
        ))
        height = int(max(
            np.linalg.norm(ordered[3] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1])
        ))

        size = max(width, height, 64)

        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (size, size))

        # Crop inner region to avoid borders
        margin = int(size * 0.1)
        if margin > 0 and size > 2 * margin:
            warped = warped[margin:-margin, margin:-margin]

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in: top-left, top-right, bottom-right, bottom-left order.
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum of coordinates: top-left has smallest, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Difference of coordinates: top-right has smallest, bottom-left has largest
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]

        return rect

    def _fallback_detection(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Fallback method when grid detection fails.

        Uses V1-style detection: color saturation + edge detection to find
        the cube region, then subdivides into 9 facelets.
        """
        height, width = image.shape[:2]

        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Threshold saturation to find colorful areas
        _, sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

        # Use coarse edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 60, 180)

        # Morphological operations to connect edges and fill the cube region
        kernel = np.ones((7, 7), np.uint8)
        edges_filled = cv2.dilate(edges, kernel, iterations=4)
        edges_filled = cv2.morphologyEx(edges_filled, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Combine saturation and edges
        combined_mask = cv2.addWeighted(sat_mask, 0.7, edges_filled, 0.3, 0)
        _, combined_mask = cv2.threshold(combined_mask, 100, 255, cv2.THRESH_BINARY)

        # Final cleanup
        kernel_large = np.ones((11, 11), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._center_crop_facelets(image)

        # Find best cube region
        min_size = min(width, height) * 0.15
        max_size = min(width, height) * 0.70
        best_region = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size * min_size:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Check squareness and saturation
            squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            roi_sat = saturation[y:y+h, x:x+w]
            avg_saturation = roi_sat.mean()

            if avg_saturation < 40:
                continue
            if squareness < 0.7:
                continue

            # Score based on squareness, saturation, and size
            score = squareness * avg_saturation * np.sqrt(area)

            if score > best_score:
                best_score = score
                # Make region square
                size = max(w, h)
                cx, cy = x + w // 2, y + h // 2
                rx = max(0, min(cx - size // 2, width - size))
                ry = max(0, min(cy - size // 2, height - size))
                best_region = image[ry:ry+size, rx:rx+size]

        if best_region is None:
            return self._center_crop_facelets(image)

        return self._subdivide_region(best_region)

    def _center_crop_facelets(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Ultimate fallback: crop center and subdivide.
        """
        height, width = image.shape[:2]
        size = int(min(width, height) * 0.6)
        x = (width - size) // 2
        y = (height - size) // 2

        region = image[y:y+size, x:x+size]
        return self._subdivide_region(region)

    def _subdivide_region(self, region: np.ndarray) -> List[np.ndarray]:
        """
        Subdivide a square region into 9 facelets.
        """
        height, width = region.shape[:2]
        cell_h = height // 3
        cell_w = width // 3

        # Add small margin to avoid borders
        margin_h = int(cell_h * 0.1)
        margin_w = int(cell_w * 0.1)

        facelets = []
        for row in range(3):
            for col in range(3):
                y1 = row * cell_h + margin_h
                y2 = (row + 1) * cell_h - margin_h
                x1 = col * cell_w + margin_w
                x2 = (col + 1) * cell_w - margin_w

                facelet = region[y1:y2, x1:x2]
                facelets.append(facelet)

        return facelets
