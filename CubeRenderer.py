"""
Rubik's Cube 3D Renderer

Provides classes for representing and rendering a Rubik's cube state
with support for move animations.

Usage:
    from CubeRenderer import CubeState, CubeRenderer, COLORS

    cube = CubeState()
    renderer = CubeRenderer(800, 800)
    frame = renderer.render_frame(cube, move="R", angle_fraction=0.5)
"""

import numpy as np
import cv2
import math

# Standard Rubik's cube colors (BGR format for OpenCV)
COLORS = {
    'W': (255, 255, 255),  # White (Up)
    'Y': (0, 255, 255),    # Yellow (Down)
    'R': (0, 0, 255),      # Red (Front)
    'O': (0, 165, 255),    # Orange (Back)
    'B': (255, 0, 0),      # Blue (Right)
    'G': (0, 255, 0),      # Green (Left)
    'K': (30, 30, 30),     # Black (internal)
}

# Face name to color mapping (solved state)
FACE_COLORS = {
    'U': 'W', 'D': 'Y', 'F': 'R', 'B': 'O', 'R': 'B', 'L': 'G'
}


class CubeState:
    """Represents the state of a Rubik's cube."""

    def __init__(self):
        """Initialize a solved cube state."""
        self.faces = {
            'U': np.full((3, 3), 'W', dtype=object),
            'D': np.full((3, 3), 'Y', dtype=object),
            'F': np.full((3, 3), 'R', dtype=object),
            'B': np.full((3, 3), 'O', dtype=object),
            'R': np.full((3, 3), 'B', dtype=object),
            'L': np.full((3, 3), 'G', dtype=object),
        }

    def apply_move(self, move):
        """Apply a move to the cube state."""
        if not move:
            return

        face = move[0]
        times = 1
        if len(move) > 1:
            if move[1] == "'":
                times = 3
            elif move[1] == "2":
                times = 2

        for _ in range(times):
            self._rotate_face_cw(face)
            self._cycle_edges_cw(face)

    def _rotate_face_cw(self, face):
        """Rotate a face clockwise."""
        self.faces[face] = np.rot90(self.faces[face], -1)

    def _cycle_edges_cw(self, face):
        """Cycle the edge pieces around a face clockwise."""
        f = self.faces
        if face == 'U':
            temp = f['F'][0].copy()
            f['F'][0] = f['R'][0].copy()
            f['R'][0] = f['B'][0].copy()
            f['B'][0] = f['L'][0].copy()
            f['L'][0] = temp
        elif face == 'D':
            temp = f['F'][2].copy()
            f['F'][2] = f['L'][2].copy()
            f['L'][2] = f['B'][2].copy()
            f['B'][2] = f['R'][2].copy()
            f['R'][2] = temp
        elif face == 'F':
            temp = f['U'][2].copy()
            f['U'][2] = f['L'][:, 2][::-1].copy()
            f['L'][:, 2] = f['D'][0].copy()
            f['D'][0] = f['R'][:, 0][::-1].copy()
            f['R'][:, 0] = temp
        elif face == 'B':
            temp = f['U'][0].copy()
            f['U'][0] = f['R'][:, 2].copy()
            f['R'][:, 2] = f['D'][2][::-1].copy()
            f['D'][2] = f['L'][:, 0].copy()
            f['L'][:, 0] = temp[::-1]
        elif face == 'R':
            temp = f['U'][:, 2].copy()
            f['U'][:, 2] = f['F'][:, 2].copy()
            f['F'][:, 2] = f['D'][:, 2].copy()
            f['D'][:, 2] = f['B'][:, 0][::-1].copy()
            f['B'][:, 0] = temp[::-1]
        elif face == 'L':
            temp = f['U'][:, 0].copy()
            f['U'][:, 0] = f['B'][:, 2][::-1].copy()
            f['B'][:, 2] = f['D'][:, 0][::-1].copy()
            f['D'][:, 0] = f['F'][:, 0].copy()
            f['F'][:, 0] = temp


class CubeRenderer:
    """Renders a 3D Rubik's cube using OpenCV."""

    def __init__(self, width=800, height=800):
        """Initialize the renderer with given dimensions."""
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.scale = min(width, height) // 5

        # Isometric-like viewing angles
        self.angle_x = math.radians(25)  # Tilt
        self.angle_y = math.radians(-45)  # Rotation

    def project_3d_to_2d(self, x, y, z):
        """Project 3D coordinates to 2D screen coordinates."""
        # Rotate around Y axis
        cos_y = math.cos(self.angle_y)
        sin_y = math.sin(self.angle_y)
        x2 = x * cos_y - z * sin_y
        z2 = x * sin_y + z * cos_y

        # Rotate around X axis
        cos_x = math.cos(self.angle_x)
        sin_x = math.sin(self.angle_x)
        y2 = y * cos_x - z2 * sin_x
        z3 = y * sin_x + z2 * cos_x

        # Simple perspective (optional)
        perspective = 1.0  # Set > 1 for perspective effect

        # Project to 2D
        screen_x = int(self.center_x + x2 * self.scale * perspective)
        screen_y = int(self.center_y - y2 * self.scale * perspective)

        return screen_x, screen_y, z3

    def rotate_point(self, point, axis, angle):
        """Rotate a 3D point around an axis."""
        x, y, z = point
        c = math.cos(angle)
        s = math.sin(angle)

        if axis == 'x':
            return (x, y * c - z * s, y * s + z * c)
        elif axis == 'y':
            return (x * c + z * s, y, -x * s + z * c)
        elif axis == 'z':
            return (x * c - y * s, x * s + y * c, z)
        return point

    def get_sticker_vertices(self, face, row, col, move_face=None, rotation_angle=0):
        """Get 3D vertices for a sticker, with optional rotation for animation."""
        # Sticker dimensions
        size = 0.85
        gap = (1 - size) / 2

        # Position within face (-1, 0, 1)
        cx = col - 1
        cy = 1 - row

        # Create sticker corners
        half = size / 2
        corners_2d = [
            (cx - half + gap, cy - half + gap),
            (cx + half - gap, cy - half + gap),
            (cx + half - gap, cy + half - gap),
            (cx - half + gap, cy + half - gap),
        ]

        # Convert to 3D based on face
        vertices = []
        for px, py in corners_2d:
            if face == 'F':
                v = (px, py, 1.5)
            elif face == 'B':
                v = (-px, py, -1.5)
            elif face == 'R':
                v = (1.5, py, -px)
            elif face == 'L':
                v = (-1.5, py, px)
            elif face == 'U':
                v = (px, 1.5, -py)
            elif face == 'D':
                v = (px, -1.5, py)
            else:
                v = (0, 0, 0)

            # Apply rotation if this sticker is on the moving layer
            if move_face and self._is_on_layer(face, row, col, move_face):
                v = self._apply_layer_rotation(v, move_face, rotation_angle)

            vertices.append(v)

        return vertices

    def _is_on_layer(self, face, row, col, move_face):
        """Check if sticker is on the rotating layer."""
        if face == move_face:
            return True

        if move_face == 'R':
            return (face in ['U', 'F', 'D'] and col == 2) or (face == 'B' and col == 0)
        elif move_face == 'L':
            return (face in ['U', 'F', 'D'] and col == 0) or (face == 'B' and col == 2)
        elif move_face == 'U':
            return face in ['F', 'R', 'B', 'L'] and row == 0
        elif move_face == 'D':
            return face in ['F', 'R', 'B', 'L'] and row == 2
        elif move_face == 'F':
            return ((face == 'U' and row == 2) or (face == 'D' and row == 0) or
                    (face == 'R' and col == 0) or (face == 'L' and col == 2))
        elif move_face == 'B':
            return ((face == 'U' and row == 0) or (face == 'D' and row == 2) or
                    (face == 'R' and col == 2) or (face == 'L' and col == 0))
        return False

    def _apply_layer_rotation(self, vertex, move_face, angle):
        """Apply rotation to a vertex based on the moving face."""
        axis_map = {'R': 'x', 'L': 'x', 'U': 'y', 'D': 'y', 'F': 'z', 'B': 'z'}
        axis = axis_map[move_face]

        # Determine rotation direction
        if move_face in ['L', 'D', 'B']:
            angle = -angle

        return self.rotate_point(vertex, axis, angle)

    def draw_sticker(self, img, vertices, color, outline_color=(0, 0, 0)):
        """Draw a filled polygon for a sticker."""
        # Project vertices to 2D
        projected = [self.project_3d_to_2d(*v) for v in vertices]

        # Get screen coordinates and depths
        points = np.array([(p[0], p[1]) for p in projected], dtype=np.int32)
        avg_depth = sum(p[2] for p in projected) / 4

        # Draw filled polygon
        cv2.fillPoly(img, [points], color)
        cv2.polylines(img, [points], True, outline_color, 2)

        return avg_depth

    def render_frame(self, cube_state, move=None, angle_fraction=0):
        """Render a single frame of the cube."""
        img = np.full((self.height, self.width, 3), 40, dtype=np.uint8)

        # Calculate rotation angle
        rotation_angle = 0
        move_face = None
        if move:
            move_face = move[0]
            total_angle = math.pi / 2
            if len(move) > 1 and move[1] == '2':
                total_angle = math.pi
            # Clockwise moves (no suffix) rotate in negative direction visually
            # Counter-clockwise (prime) moves rotate in positive direction
            direction = 1 if (len(move) > 1 and move[1] == "'") else -1
            rotation_angle = direction * total_angle * angle_fraction

        # Collect all stickers with their depths
        stickers = []
        for face in ['U', 'D', 'F', 'B', 'R', 'L']:
            for row in range(3):
                for col in range(3):
                    vertices = self.get_sticker_vertices(face, row, col, move_face, rotation_angle)
                    color_code = cube_state.faces[face][row, col]
                    color = COLORS[color_code]

                    # Calculate average depth for sorting
                    projected = [self.project_3d_to_2d(*v) for v in vertices]
                    avg_depth = sum(p[2] for p in projected) / 4

                    stickers.append((avg_depth, vertices, color))

        # Sort by depth (draw back to front)
        stickers.sort(key=lambda x: x[0])

        # Draw stickers
        for _, vertices, color in stickers:
            self.draw_sticker(img, vertices, color)

        # Add move label
        if move:
            label = f"Move: {move}"
            cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Progress bar
            bar_width = 200
            bar_height = 20
            bar_x = 20
            bar_y = self.height - 40
            progress = int(bar_width * angle_fraction)

            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                          (100, 100, 100), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height),
                          (0, 255, 0), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                          (255, 255, 255), 2)

            pct_text = f"{int(angle_fraction * 100)}%"
            cv2.putText(img, pct_text, (bar_x + bar_width + 10, bar_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return img
