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

import sys
import os
import argparse
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CubeRenderer import CubeState, CubeRenderer, COLORS, FACE_COLORS


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
