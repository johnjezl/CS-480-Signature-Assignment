#!/usr/bin/env python3
"""
CLI Visualizer for Rubik's Cube Facelets

Visualizes facelets from facelets_for_debug/ directory as a colored
cube net in the terminal. Uses ANSI colors for display.

Usage:
    python visualize_facelets.py                    # Visualize default facelets
    python visualize_facelets.py --path Black       # Visualize Black subfolder
    python visualize_facelets.py --no-color         # ASCII only (no colors)
    python visualize_facelets.py --confidence       # Show confidence values
"""

import os
import sys
import argparse
import cv2
import numpy as np

# ANSI color codes for terminal display
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Background colors (bright versions for visibility)
    BG_WHITE = '\033[107m'      # Bright white background
    BG_YELLOW = '\033[103m'     # Bright yellow background
    BG_RED = '\033[101m'        # Bright red background
    BG_ORANGE = '\033[48;5;208m'  # Orange (256 color mode)
    BG_BLUE = '\033[104m'       # Bright blue background
    BG_GREEN = '\033[102m'      # Bright green background
    BG_BLACK = '\033[40m'       # Black background (for unknown)

    # Foreground colors for contrast
    FG_BLACK = '\033[30m'
    FG_WHITE = '\033[97m'

    @classmethod
    def get_color(cls, color_name):
        """Get background color code for a given color name."""
        color_map = {
            'white': (cls.BG_WHITE, cls.FG_BLACK),
            'yellow': (cls.BG_YELLOW, cls.FG_BLACK),
            'red': (cls.BG_RED, cls.FG_WHITE),
            'orange': (cls.BG_ORANGE, cls.FG_BLACK),
            'blue': (cls.BG_BLUE, cls.FG_WHITE),
            'green': (cls.BG_GREEN, cls.FG_BLACK),
        }
        return color_map.get(color_name.lower(), (cls.BG_BLACK, cls.FG_WHITE))


# Color letter mapping
COLOR_LETTERS = {
    'white': 'W',
    'yellow': 'Y',
    'red': 'R',
    'orange': 'O',
    'blue': 'B',
    'green': 'G',
    'unknown': '?'
}

# Standard face names
FACES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FRONT', 'BACK']

# Facelet numbering reference:
# 0 | 1 | 2
# ---------
# 3 | 4 | 5
# ---------
# 6 | 7 | 8


def load_facelets(base_path, face_name):
    """
    Load all 9 facelets for a given face.

    Args:
        base_path: Base directory containing face folders
        face_name: Name of the face (UP, DOWN, etc.)

    Returns:
        List of 9 facelet images (64x64 BGR) or None for missing
    """
    face_path = os.path.join(base_path, face_name)
    facelets = []

    for i in range(9):
        facelet_path = os.path.join(face_path, f'facelet_{i}.png')
        if os.path.exists(facelet_path):
            img = cv2.imread(facelet_path)
            facelets.append(img)
        else:
            facelets.append(None)

    return facelets


def classify_facelets(facelets, classifier):
    """
    Classify all facelets using the CNN classifier.

    Args:
        facelets: List of 9 facelet images
        classifier: FaceletColorClassifier instance

    Returns:
        List of 9 tuples (color, confidence) or ('unknown', 0) for missing
    """
    results = []
    for facelet in facelets:
        if facelet is not None:
            try:
                color, conf = classifier.classify_facelet(facelet)
                results.append((color, conf))
            except Exception as e:
                results.append(('unknown', 0))
        else:
            results.append(('unknown', 0))
    return results


def format_cell(color, confidence, use_color=True, show_confidence=False):
    """
    Format a single cell for display.

    Args:
        color: Color name (e.g., 'white', 'red')
        confidence: Confidence percentage
        use_color: Whether to use ANSI colors
        show_confidence: Whether to show confidence value

    Returns:
        Formatted string for the cell
    """
    letter = COLOR_LETTERS.get(color, '?')

    if show_confidence:
        # Show letter with confidence (e.g., "R 95")
        cell_content = f" {letter}{int(confidence):3d}"
    else:
        # Just show the letter centered
        cell_content = f"  {letter}  "

    if use_color:
        bg, fg = Colors.get_color(color)
        return f"{bg}{fg}{cell_content}{Colors.RESET}"
    else:
        return cell_content


def render_face_row(classifications, row, use_color=True, show_confidence=False):
    """
    Render one row (3 cells) of a face.

    Args:
        classifications: List of 9 (color, confidence) tuples
        row: Row index (0, 1, or 2)
        use_color: Whether to use ANSI colors
        show_confidence: Whether to show confidence

    Returns:
        Formatted string for the row
    """
    start_idx = row * 3
    cells = []
    for i in range(3):
        color, conf = classifications[start_idx + i]
        cells.append(format_cell(color, conf, use_color, show_confidence))

    separator = "|" if not use_color else "|"
    return separator + separator.join(cells) + separator


def print_cube_net(all_classifications, use_color=True, show_confidence=False):
    """
    Print the cube as an unfolded net.

    Layout:
              +-------+
              |   U   |
              |  UP   |
              +-------+
    +-------+-------+-------+-------+
    |   L   |   F   |   R   |   B   |
    | LEFT  | FRONT | RIGHT | BACK  |
    +-------+-------+-------+-------+
              +-------+
              |   D   |
              | DOWN  |
              +-------+

    Args:
        all_classifications: Dict mapping face name to list of 9 classifications
        use_color: Whether to use ANSI colors
        show_confidence: Whether to show confidence values
    """
    # Cell width depends on whether we show confidence
    cell_width = 5 if show_confidence else 5
    face_width = cell_width * 3 + 4  # 3 cells + separators

    # Helper to create horizontal border
    def h_border(count=1):
        single = "+" + "-" * cell_width
        return (single * 3 + "+") * count

    # Helper to get empty space of face width
    def empty_face():
        return " " * face_width

    # Print UP face (centered above)
    print()
    title = "Rubik's Cube Visualizer"
    subtitle = "(from facelets_for_debug)"
    total_width = face_width * 4
    print(f"{title:^{total_width}}")
    print(f"{subtitle:^{total_width}}")
    print()

    # UP face header
    indent = empty_face()
    print(indent + h_border())
    print(indent + f"|{'UP':^{face_width-2}}|")
    print(indent + h_border())

    # UP face content
    up_class = all_classifications.get('UP', [('unknown', 0)] * 9)
    for row in range(3):
        print(indent + render_face_row(up_class, row, use_color, show_confidence))
    print(indent + h_border())

    print()

    # Middle row: LEFT, FRONT, RIGHT, BACK
    middle_faces = ['LEFT', 'FRONT', 'RIGHT', 'BACK']

    # Headers
    header_row = ""
    for face in middle_faces:
        header_row += h_border()
    print(header_row)

    label_row = ""
    for face in middle_faces:
        label_row += f"|{face:^{face_width-2}}|"
    print(label_row)

    border_row = ""
    for _ in middle_faces:
        border_row += h_border()
    print(border_row)

    # Content rows
    for row in range(3):
        content_row = ""
        for face in middle_faces:
            face_class = all_classifications.get(face, [('unknown', 0)] * 9)
            content_row += render_face_row(face_class, row, use_color, show_confidence)
        print(content_row)

    print(border_row)

    print()

    # DOWN face (centered below)
    print(indent + h_border())
    print(indent + f"|{'DOWN':^{face_width-2}}|")
    print(indent + h_border())

    down_class = all_classifications.get('DOWN', [('unknown', 0)] * 9)
    for row in range(3):
        print(indent + render_face_row(down_class, row, use_color, show_confidence))
    print(indent + h_border())
    print()


def print_individual_faces(all_classifications, use_color=True, show_confidence=False):
    """
    Print each face individually with its name.
    """
    cell_width = 5
    face_width = cell_width * 3 + 4

    def h_border():
        return ("+" + "-" * cell_width) * 3 + "+"

    print("\nIndividual Faces:")
    print("=" * 60)

    for face_name in FACES:
        face_class = all_classifications.get(face_name, [('unknown', 0)] * 9)

        print(f"\n{face_name}:")
        print(h_border())
        for row in range(3):
            print(render_face_row(face_class, row, use_color, show_confidence))
        print(h_border())


def print_legend(use_color=True):
    """Print a color legend."""
    print("\nLegend:")
    colors = ['white', 'yellow', 'red', 'orange', 'blue', 'green']
    legend_items = []

    for color in colors:
        letter = COLOR_LETTERS[color]
        if use_color:
            bg, fg = Colors.get_color(color)
            legend_items.append(f"{bg}{fg} {letter} {Colors.RESET} = {color.capitalize()}")
        else:
            legend_items.append(f"{letter} = {color.capitalize()}")

    print("  " + "  |  ".join(legend_items[:3]))
    print("  " + "  |  ".join(legend_items[3:]))
    print()


def print_facelet_numbering():
    """Print the facelet numbering reference."""
    print("\nFacelet Numbering (row-major):")
    print("  +---+---+---+")
    print("  | 0 | 1 | 2 |")
    print("  +---+---+---+")
    print("  | 3 | 4 | 5 |")
    print("  +---+---+---+")
    print("  | 6 | 7 | 8 |")
    print("  +---+---+---+")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Rubik\'s Cube facelets from debug images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_facelets.py                    # Default visualization
  python visualize_facelets.py --path Black       # Visualize Black subfolder
  python visualize_facelets.py --no-color         # ASCII only
  python visualize_facelets.py --confidence       # Show confidence values
  python visualize_facelets.py --individual       # Show each face separately
        """
    )

    parser.add_argument('--path', type=str, default='',
                        help='Subfolder path within facelets_for_debug (e.g., "Black")')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable ANSI colors (ASCII only)')
    parser.add_argument('--confidence', action='store_true',
                        help='Show confidence percentages')
    parser.add_argument('--individual', action='store_true',
                        help='Also show each face individually')
    parser.add_argument('--legend', action='store_true',
                        help='Show color legend')
    parser.add_argument('--numbering', action='store_true',
                        help='Show facelet numbering reference')

    args = parser.parse_args()

    # Determine base path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'facelets_for_debug')

    if args.path:
        base_path = os.path.join(base_path, args.path)

    if not os.path.exists(base_path):
        print(f"Error: Path not found: {base_path}")
        sys.exit(1)

    # Check if any face folders exist
    available_faces = [f for f in FACES if os.path.exists(os.path.join(base_path, f))]
    if not available_faces:
        print(f"Error: No face folders found in {base_path}")
        print(f"Expected folders: {', '.join(FACES)}")
        sys.exit(1)

    print(f"Loading facelets from: {base_path}")
    print(f"Found faces: {', '.join(available_faces)}")

    # Load classifier
    try:
        from FaceletColorClassifier import FaceletColorClassifier
        classifier = FaceletColorClassifier()
        print(f"Classifier loaded on: {classifier.device}")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        sys.exit(1)

    # Load and classify all faces
    all_classifications = {}

    for face_name in FACES:
        face_path = os.path.join(base_path, face_name)
        if os.path.exists(face_path):
            facelets = load_facelets(base_path, face_name)
            classifications = classify_facelets(facelets, classifier)
            all_classifications[face_name] = classifications

            # Count classified facelets
            valid = sum(1 for c, _ in classifications if c != 'unknown')
            print(f"  {face_name}: {valid}/9 facelets classified")

    use_color = not args.no_color

    # Print visualizations
    if args.numbering:
        print_facelet_numbering()

    print_cube_net(all_classifications, use_color, args.confidence)

    if args.individual:
        print_individual_faces(all_classifications, use_color, args.confidence)

    if args.legend:
        print_legend(use_color)

    # Print summary
    print("\nSummary:")
    total_facelets = 0
    total_classified = 0
    color_counts = {c: 0 for c in COLOR_LETTERS.keys()}

    for face_name, classifications in all_classifications.items():
        for color, conf in classifications:
            total_facelets += 1
            if color != 'unknown':
                total_classified += 1
                color_counts[color] = color_counts.get(color, 0) + 1

    print(f"  Total facelets: {total_facelets}")
    print(f"  Classified: {total_classified}")

    if total_classified > 0:
        print("  Color distribution:")
        for color in ['white', 'yellow', 'red', 'orange', 'blue', 'green']:
            count = color_counts.get(color, 0)
            if count > 0:
                bar = '#' * count
                print(f"    {color.capitalize():8s}: {count:2d} {bar}")


if __name__ == '__main__':
    main()
