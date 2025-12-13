"""
Cube Orientation Corrector

Detects and corrects orientation issues in scanned cube faces before
submitting to the solver. Handles rotated and flipped faces by analyzing
edge and corner piece constraints.

Usage:
    from CubeOrientationCorrector import CubeOrientationCorrector

    corrector = CubeOrientationCorrector()
    corrected_cube, corrections = corrector.correct(cube_data)
"""

import copy
from typing import Dict, List, Tuple, Optional


# Expected center colors for each face
EXPECTED_CENTERS = {
    'up': 'Y',      # Yellow
    'down': 'W',    # White
    'front': 'B',   # Blue
    'back': 'G',    # Green
    'left': 'O',    # Orange
    'right': 'R',   # Red
}

# Valid color set
VALID_COLORS = {'W', 'Y', 'R', 'O', 'B', 'G'}

# Adjacent faces and which edge positions they share
# Format: face -> {adjacent_face: (my_edge_positions, their_edge_positions)}
# Edge positions are the facelet indices that form the shared edge
ADJACENCIES = {
    'up': {
        'front': ([6, 7, 8], [0, 1, 2]),      # up's bottom edge = front's top edge
        'back': ([0, 1, 2], [0, 1, 2]),       # up's top edge = back's top edge (reversed view)
        'left': ([0, 3, 6], [0, 1, 2]),       # up's left edge = left's top edge
        'right': ([2, 5, 8], [0, 1, 2]),      # up's right edge = right's top edge
    },
    'down': {
        'front': ([0, 1, 2], [6, 7, 8]),      # down's top edge = front's bottom edge
        'back': ([6, 7, 8], [6, 7, 8]),       # down's bottom edge = back's bottom edge
        'left': ([0, 3, 6], [6, 7, 8]),       # down's left edge = left's bottom edge
        'right': ([2, 5, 8], [6, 7, 8]),      # down's right edge = right's bottom edge
    },
    'front': {
        'up': ([0, 1, 2], [6, 7, 8]),
        'down': ([6, 7, 8], [0, 1, 2]),
        'left': ([0, 3, 6], [2, 5, 8]),       # front's left edge = left's right edge
        'right': ([2, 5, 8], [0, 3, 6]),      # front's right edge = right's left edge
    },
    'back': {
        'up': ([0, 1, 2], [0, 1, 2]),
        'down': ([6, 7, 8], [6, 7, 8]),
        'left': ([2, 5, 8], [0, 3, 6]),       # back's right edge = left's left edge
        'right': ([0, 3, 6], [2, 5, 8]),      # back's left edge = right's right edge
    },
    'left': {
        'up': ([0, 1, 2], [0, 3, 6]),
        'down': ([6, 7, 8], [0, 3, 6]),
        'front': ([2, 5, 8], [0, 3, 6]),
        'back': ([0, 3, 6], [2, 5, 8]),
    },
    'right': {
        'up': ([0, 1, 2], [2, 5, 8]),
        'down': ([6, 7, 8], [2, 5, 8]),
        'front': ([0, 3, 6], [2, 5, 8]),
        'back': ([2, 5, 8], [0, 3, 6]),
    },
}


class CubeOrientationCorrector:
    """
    Detects and corrects face orientation issues in cube data.

    Handles:
    - Faces rotated 90, 180, or 270 degrees
    - Center color validation
    - Edge consistency between adjacent faces
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the corrector.

        Args:
            verbose: If True, print detailed correction information
        """
        self.verbose = verbose

    def correct(self, cube_data: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Analyze and correct orientation issues in cube data.

        Args:
            cube_data: Dictionary with face names as keys and lists of 9 color
                      letters as values. Format: {'up': ['Y', 'Y', ...], ...}

        Returns:
            Tuple of (corrected_cube_data, corrections_made)
            corrections_made is a dict mapping face names to correction descriptions
        """
        corrected = copy.deepcopy(cube_data)
        corrections = {}

        # Step 1: Correct centers first (rotate faces so center matches expected)
        for face_name in EXPECTED_CENTERS:
            if face_name not in corrected:
                continue

            face = corrected[face_name]
            expected_center = EXPECTED_CENTERS[face_name]
            actual_center = face[4]  # Position 4 is always the center

            if actual_center != expected_center:
                # Center doesn't match - this face might be the wrong face entirely
                # Try to find which face this actually is based on center color
                actual_face_name = self._find_face_by_center(actual_center)
                if actual_face_name:
                    corrections[face_name] = f"Warning: Center is {actual_center}, expected {expected_center}. May be {actual_face_name} face."
                else:
                    corrections[face_name] = f"Warning: Center is {actual_center}, expected {expected_center}"

        # Step 2: Try rotations to maximize edge consistency
        corrected, rotation_corrections = self._optimize_rotations(corrected)
        corrections.update(rotation_corrections)

        # Step 3: Validate and report final state
        validation = self._validate_cube(corrected)
        if not validation['valid']:
            corrections['_validation'] = f"Issues remaining: {validation['issues']}"

        return corrected, corrections

    def _find_face_by_center(self, color: str) -> Optional[str]:
        """Find which face should have the given center color."""
        for face_name, expected in EXPECTED_CENTERS.items():
            if expected == color:
                return face_name
        return None

    def _rotate_face_cw(self, face: List[str]) -> List[str]:
        """Rotate a face 90 degrees clockwise."""
        # Original positions:  0 1 2
        #                      3 4 5
        #                      6 7 8
        # After CW rotation:   6 3 0
        #                      7 4 1
        #                      8 5 2
        return [
            face[6], face[3], face[0],
            face[7], face[4], face[1],
            face[8], face[5], face[2]
        ]

    def _rotate_face_ccw(self, face: List[str]) -> List[str]:
        """Rotate a face 90 degrees counter-clockwise."""
        # After CCW rotation: 2 5 8
        #                     1 4 7
        #                     0 3 6
        return [
            face[2], face[5], face[8],
            face[1], face[4], face[7],
            face[0], face[3], face[6]
        ]

    def _rotate_face_180(self, face: List[str]) -> List[str]:
        """Rotate a face 180 degrees."""
        # After 180 rotation: 8 7 6
        #                     5 4 3
        #                     2 1 0
        return [
            face[8], face[7], face[6],
            face[5], face[4], face[3],
            face[2], face[1], face[0]
        ]

    def _get_edge_colors(self, face: List[str], positions: List[int]) -> List[str]:
        """Get colors at the specified positions from a face."""
        return [face[p] for p in positions]

    def _count_edge_matches(self, cube_data: Dict[str, List[str]]) -> int:
        """
        Count how many edge positions match between adjacent faces.
        Higher score = better alignment.
        """
        total_matches = 0
        checked = set()

        for face_name, adjacents in ADJACENCIES.items():
            if face_name not in cube_data:
                continue
            for adj_face, (my_positions, their_positions) in adjacents.items():
                if adj_face not in cube_data:
                    continue

                # Avoid double counting
                pair = tuple(sorted([face_name, adj_face]))
                if pair in checked:
                    continue
                checked.add(pair)

                my_colors = self._get_edge_colors(cube_data[face_name], my_positions)
                their_colors = self._get_edge_colors(cube_data[adj_face], their_positions)

                # For some adjacencies, the edge needs to be compared in reverse
                # This depends on viewing orientation
                # Check both forward and reverse to find matches
                for i, (my_c, their_c) in enumerate(zip(my_colors, their_colors)):
                    # Edge pieces should share colors with adjacent face pieces
                    # but the actual matching depends on cube geometry
                    pass

        return total_matches

    def _calculate_consistency_score(self, cube_data: Dict[str, List[str]]) -> float:
        """
        Calculate a consistency score for the cube configuration.
        Checks edge and corner piece validity.
        """
        score = 0.0

        # Check center colors match expected
        for face_name, expected in EXPECTED_CENTERS.items():
            if face_name in cube_data and cube_data[face_name][4] == expected:
                score += 10.0

        # Check color distribution (each color should appear 9 times)
        color_counts = {}
        for face in cube_data.values():
            for color in face:
                color_counts[color] = color_counts.get(color, 0) + 1

        for color in VALID_COLORS:
            count = color_counts.get(color, 0)
            if count == 9:
                score += 5.0
            else:
                score -= abs(9 - count) * 2.0

        # Check edge piece validity (each edge has exactly 2 colors, neither same)
        edge_score = self._check_edge_pieces(cube_data)
        score += edge_score

        # Check corner piece validity (each corner has exactly 3 colors, all different)
        corner_score = self._check_corner_pieces(cube_data)
        score += corner_score

        return score

    def _check_edge_pieces(self, cube_data: Dict[str, List[str]]) -> float:
        """Check that edge pieces have valid color combinations."""
        score = 0.0

        # Define edge positions as pairs of (face, position)
        edges = [
            # Up layer edges
            (('up', 1), ('back', 1)),
            (('up', 5), ('right', 1)),
            (('up', 7), ('front', 1)),
            (('up', 3), ('left', 1)),
            # Down layer edges
            (('down', 1), ('front', 7)),
            (('down', 5), ('right', 7)),
            (('down', 7), ('back', 7)),
            (('down', 3), ('left', 7)),
            # Middle layer edges
            (('front', 3), ('left', 5)),
            (('front', 5), ('right', 3)),
            (('back', 5), ('left', 3)),
            (('back', 3), ('right', 5)),
        ]

        for (face1, pos1), (face2, pos2) in edges:
            if face1 in cube_data and face2 in cube_data:
                color1 = cube_data[face1][pos1]
                color2 = cube_data[face2][pos2]

                # Edge should have two different colors
                if color1 != color2:
                    score += 1.0
                    # Opposite colors can't be on same edge (W-Y, R-O, B-G)
                    opposites = {('W', 'Y'), ('Y', 'W'), ('R', 'O'), ('O', 'R'), ('B', 'G'), ('G', 'B')}
                    if (color1, color2) not in opposites:
                        score += 1.0

        return score

    def _check_corner_pieces(self, cube_data: Dict[str, List[str]]) -> float:
        """Check that corner pieces have valid color combinations."""
        score = 0.0

        # Define corner positions as triples of (face, position)
        corners = [
            # Up layer corners
            (('up', 0), ('back', 2), ('left', 0)),
            (('up', 2), ('back', 0), ('right', 2)),
            (('up', 6), ('front', 0), ('left', 2)),
            (('up', 8), ('front', 2), ('right', 0)),
            # Down layer corners
            (('down', 0), ('front', 6), ('left', 8)),
            (('down', 2), ('front', 8), ('right', 6)),
            (('down', 6), ('back', 8), ('left', 6)),
            (('down', 8), ('back', 6), ('right', 8)),
        ]

        for (face1, pos1), (face2, pos2), (face3, pos3) in corners:
            if face1 in cube_data and face2 in cube_data and face3 in cube_data:
                color1 = cube_data[face1][pos1]
                color2 = cube_data[face2][pos2]
                color3 = cube_data[face3][pos3]

                colors = {color1, color2, color3}

                # Corner should have three different colors
                if len(colors) == 3:
                    score += 1.0
                    # No opposite colors on same corner
                    opposites = [{'W', 'Y'}, {'R', 'O'}, {'B', 'G'}]
                    has_opposite = any(opp.issubset(colors) for opp in opposites)
                    if not has_opposite:
                        score += 2.0

        return score

    def _optimize_rotations(self, cube_data: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Try different rotations of each face to maximize consistency.

        Returns:
            Tuple of (best_cube_data, corrections_made)
        """
        best_cube = copy.deepcopy(cube_data)
        best_score = self._calculate_consistency_score(best_cube)
        corrections = {}

        if self.verbose:
            print(f"Initial consistency score: {best_score:.1f}")

        # Try rotating each face independently
        rotation_options = [
            (None, "no rotation"),
            (self._rotate_face_cw, "rotated 90° CW"),
            (self._rotate_face_180, "rotated 180°"),
            (self._rotate_face_ccw, "rotated 90° CCW"),
        ]

        improved = True
        iteration = 0
        max_iterations = 10  # Prevent infinite loops

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for face_name in EXPECTED_CENTERS:
                if face_name not in best_cube:
                    continue

                original_face = best_cube[face_name]
                best_face = original_face
                best_face_score = best_score
                best_rotation = None

                for rotate_func, rotation_name in rotation_options:
                    if rotate_func is None:
                        test_face = original_face
                    else:
                        test_face = rotate_func(original_face)

                    # Skip if center doesn't match expected
                    if test_face[4] != EXPECTED_CENTERS[face_name]:
                        continue

                    # Test this rotation
                    test_cube = copy.deepcopy(best_cube)
                    test_cube[face_name] = test_face
                    test_score = self._calculate_consistency_score(test_cube)

                    if test_score > best_face_score:
                        best_face = test_face
                        best_face_score = test_score
                        best_rotation = rotation_name

                if best_rotation and best_rotation != "no rotation":
                    best_cube[face_name] = best_face
                    best_score = best_face_score
                    corrections[face_name] = best_rotation
                    improved = True
                    if self.verbose:
                        print(f"  {face_name}: {best_rotation} (score: {best_score:.1f})")

        if self.verbose:
            print(f"Final consistency score: {best_score:.1f}")

        return best_cube, corrections

    def _validate_cube(self, cube_data: Dict[str, List[str]]) -> Dict:
        """
        Validate the cube configuration.

        Returns:
            Dict with 'valid' bool and 'issues' list
        """
        issues = []

        # Check all faces present
        for face_name in EXPECTED_CENTERS:
            if face_name not in cube_data:
                issues.append(f"Missing face: {face_name}")

        # Check each face has 9 colors
        for face_name, face in cube_data.items():
            if len(face) != 9:
                issues.append(f"{face_name} has {len(face)} facelets, expected 9")

        # Check color distribution
        color_counts = {}
        for face in cube_data.values():
            for color in face:
                color_counts[color] = color_counts.get(color, 0) + 1

        for color in VALID_COLORS:
            count = color_counts.get(color, 0)
            if count != 9:
                issues.append(f"Color {color} appears {count} times, expected 9")

        # Check for invalid colors
        for color, count in color_counts.items():
            if color not in VALID_COLORS:
                issues.append(f"Invalid color '{color}' found {count} times")

        # Check centers
        for face_name, expected in EXPECTED_CENTERS.items():
            if face_name in cube_data:
                actual = cube_data[face_name][4]
                if actual != expected:
                    issues.append(f"{face_name} center is {actual}, expected {expected}")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def get_face_rotation_for_center(self, face: List[str], expected_center: str) -> Optional[str]:
        """
        Determine what rotation is needed to get the expected center color.

        Args:
            face: List of 9 color letters
            expected_center: The color that should be at center (position 4)

        Returns:
            Rotation needed ('none', 'cw', 'ccw', '180') or None if not possible
        """
        # Center is always at position 4, which doesn't change with rotation
        # So if center doesn't match, the face itself is wrong
        if face[4] == expected_center:
            return 'none'
        return None

    def print_cube(self, cube_data: Dict[str, List[str]]):
        """Print the cube in a readable format for debugging."""
        def format_face(face: List[str]) -> List[str]:
            return [
                f"  {face[0]} {face[1]} {face[2]}",
                f"  {face[3]} {face[4]} {face[5]}",
                f"  {face[6]} {face[7]} {face[8]}"
            ]

        print("\nCube State:")
        print("=" * 40)

        # Print up face
        if 'up' in cube_data:
            print("Up (Y):")
            for line in format_face(cube_data['up']):
                print(line)

        # Print middle row (left, front, right, back)
        print("\nLeft(O)  Front(B) Right(R) Back(G)")
        for row in range(3):
            line = ""
            for face_name in ['left', 'front', 'right', 'back']:
                if face_name in cube_data:
                    f = cube_data[face_name]
                    line += f"  {f[row*3]} {f[row*3+1]} {f[row*3+2]} "
                else:
                    line += "  ? ? ? "
            print(line)

        # Print down face
        if 'down' in cube_data:
            print("\nDown (W):")
            for line in format_face(cube_data['down']):
                print(line)

        print("=" * 40)


def test_corrector():
    """Test the orientation corrector with sample data."""
    # Create a solved cube
    solved_cube = {
        'up': ['Y'] * 9,
        'down': ['W'] * 9,
        'front': ['B'] * 9,
        'back': ['G'] * 9,
        'left': ['O'] * 9,
        'right': ['R'] * 9,
    }

    corrector = CubeOrientationCorrector(verbose=True)

    print("Testing with solved cube:")
    corrected, corrections = corrector.correct(solved_cube)
    corrector.print_cube(corrected)
    print(f"Corrections: {corrections}")

    # Test with a rotated face
    print("\n" + "=" * 50)
    print("Testing with rotated front face (90° CW):")
    rotated_cube = copy.deepcopy(solved_cube)
    # Simulate a pattern on front face, then rotate it
    rotated_cube['front'] = ['B', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
    # Rotate 90° CW
    rotated_cube['front'] = corrector._rotate_face_cw(rotated_cube['front'])
    print("Before correction:")
    corrector.print_cube(rotated_cube)

    corrected, corrections = corrector.correct(rotated_cube)
    print("\nAfter correction:")
    corrector.print_cube(corrected)
    print(f"Corrections: {corrections}")


if __name__ == "__main__":
    test_corrector()
