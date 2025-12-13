#!/usr/bin/env python3
"""
Rubik's Cube Move Animator

Generates and displays animation frames showing a single cube move rotation.
Uses OpenCV for rendering (compatible with Jetson).

Usage:
    python tools/cube_move_animator.py R      # Show R move animation
    python tools/cube_move_animator.py U'     # Show U' (counter-clockwise) move
    python tools/cube_move_animator.py F2     # Show F2 (180 degree) move
"""

import numpy as np
import cv2
import argparse
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
        self.faces[face] = np.rot90(self.faces[face], -1)

    def _cycle_edges_cw(self, face):
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
            direction = -1 if (len(move) > 1 and move[1] == "'") else 1
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


def animate_move(move, num_frames=30, delay_ms=30):
    """Animate a cube move and display on screen."""
    if not move or move[0] not in 'UDFBRL':
        print(f"Invalid move: {move}")
        print("Valid moves: U, D, F, B, R, L (optionally followed by ' or 2)")
        return

    print(f"Animating move: {move}")
    print("Press any key to exit, or wait for animation to complete")

    cube = CubeState()
    renderer = CubeRenderer(800, 800)

    window_name = f"Cube Move: {move}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Animation loop
    for i in range(num_frames + 1):
        angle_fraction = i / num_frames

        # Use easing for smoother animation
        # ease_fraction = 0.5 - 0.5 * math.cos(math.pi * angle_fraction)
        ease_fraction = angle_fraction  # Linear for now

        frame = renderer.render_frame(cube, move, ease_fraction)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay_ms)
        if key != -1:
            break

    # Show final state
    cube.apply_move(move)
    final_frame = renderer.render_frame(cube, move, 1.0)

    # Add "Complete" text
    cv2.putText(final_frame, "Complete! Press any key to exit",
                (150, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(window_name, final_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Animate a Rubik's cube move",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cube_move_animator.py R       # Right face clockwise
    python cube_move_animator.py U'      # Up face counter-clockwise
    python cube_move_animator.py F2      # Front face 180 degrees
    python cube_move_animator.py L       # Left face clockwise

Move Notation:
    U/D/F/B/R/L  - Face to rotate (Up/Down/Front/Back/Right/Left)
    (nothing)    - Clockwise 90 degrees
    '            - Counter-clockwise 90 degrees
    2            - 180 degrees
        """
    )

    parser.add_argument('move', nargs='?', default='R',
                        help="Move to animate (default: R)")
    parser.add_argument('--frames', type=int, default=30,
                        help="Number of animation frames (default: 30)")
    parser.add_argument('--delay', type=int, default=30,
                        help="Delay between frames in ms (default: 30)")

    args = parser.parse_args()

    # Normalize move notation
    move = args.move.upper().replace("'", "'")

    animate_move(move, num_frames=args.frames, delay_ms=args.delay)


if __name__ == '__main__':
    main()
