"""
Generate test images for the Rubik's cube facelet segmenter.
Uses PPM format (no external dependencies required).

Run this script to create test_images/ directory with sample cube face images.
"""

import os
import struct


# Rubik's cube standard colors (RGB)
COLORS = {
    'W': (255, 255, 255),  # White
    'Y': (255, 255, 0),    # Yellow
    'R': (255, 0, 0),      # Red
    'O': (255, 165, 0),    # Orange
    'B': (0, 0, 255),      # Blue
    'G': (0, 255, 0),      # Green
}

# Test face patterns (9 colors each, top-left to bottom-right)
TEST_PATTERNS = {
    'white_face': ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
    'red_face': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'],
    'scrambled_1': ['R', 'B', 'Y', 'G', 'W', 'O', 'Y', 'R', 'B'],
    'scrambled_2': ['W', 'O', 'G', 'R', 'B', 'Y', 'O', 'W', 'R'],
    'checkerboard': ['R', 'W', 'R', 'W', 'R', 'W', 'R', 'W', 'R'],
    'rainbow': ['R', 'O', 'Y', 'G', 'B', 'W', 'R', 'O', 'Y'],
}


def write_ppm(filename: str, width: int, height: int, pixels: list):
    """
    Write an image in PPM format (P6 binary).

    Args:
        filename: Output filename
        width: Image width
        height: Image height
        pixels: List of (r, g, b) tuples, row by row
    """
    with open(filename, 'wb') as f:
        # PPM header
        header = f"P6\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))

        # Pixel data
        for r, g, b in pixels:
            f.write(struct.pack('BBB', r, g, b))


def create_cube_face_image(
    pattern: list,
    image_size: tuple = (640, 480),
    face_size: int = 300,
    face_offset: tuple = None,
    background: tuple = (50, 50, 50),
    gap: int = 4
) -> list:
    """
    Create a synthetic Rubik's cube face image.

    Args:
        pattern: List of 9 color codes ('R', 'W', etc.)
        image_size: (width, height) of output image
        face_size: Size of cube face in pixels
        face_offset: (x, y) offset of face, or None for centered
        background: Background RGB color
        gap: Gap between facelets in pixels

    Returns:
        List of (r, g, b) tuples for each pixel
    """
    width, height = image_size

    # Center the face if no offset specified
    if face_offset is None:
        face_offset = ((width - face_size) // 2, (height - face_size) // 2)

    fx, fy = face_offset
    facelet_size = face_size // 3

    # Initialize with background color
    pixels = [background] * (width * height)

    # Draw each facelet
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            color = COLORS[pattern[idx]]

            # Facelet bounds (with gap)
            x_start = fx + col * facelet_size + gap
            x_end = fx + (col + 1) * facelet_size - gap
            y_start = fy + row * facelet_size + gap
            y_end = fy + (row + 1) * facelet_size - gap

            # Fill facelet pixels
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if 0 <= x < width and 0 <= y < height:
                        pixels[y * width + x] = color

    # Draw black border around each facelet
    border_color = (0, 0, 0)
    for row in range(3):
        for col in range(3):
            x_start = fx + col * facelet_size + gap
            x_end = fx + (col + 1) * facelet_size - gap
            y_start = fy + row * facelet_size + gap
            y_end = fy + (row + 1) * facelet_size - gap

            # Top and bottom borders
            for x in range(x_start, x_end):
                for t in range(2):  # 2-pixel border
                    if 0 <= x < width:
                        if 0 <= y_start + t < height:
                            pixels[(y_start + t) * width + x] = border_color
                        if 0 <= y_end - 1 - t < height:
                            pixels[(y_end - 1 - t) * width + x] = border_color

            # Left and right borders
            for y in range(y_start, y_end):
                for t in range(2):
                    if 0 <= y < height:
                        if 0 <= x_start + t < width:
                            pixels[y * width + x_start + t] = border_color
                        if 0 <= x_end - 1 - t < width:
                            pixels[y * width + x_end - 1 - t] = border_color

    return pixels


def create_expected_facelets(pattern: list, size: int = 64) -> list:
    """
    Create the expected 64x64 facelet images for a pattern.

    Args:
        pattern: List of 9 color codes
        size: Output size for each facelet

    Returns:
        List of 9 pixel lists, each for a size x size image
    """
    facelets = []

    for color_code in pattern:
        color = COLORS[color_code]
        # Solid color square with thin border
        pixels = []
        border = 2
        for y in range(size):
            for x in range(size):
                if x < border or x >= size - border or y < border or y >= size - border:
                    pixels.append((0, 0, 0))  # Black border
                else:
                    pixels.append(color)
        facelets.append(pixels)

    return facelets


def generate_all_test_images(base_dir: str = "test_images"):
    """Generate all test images and expected outputs."""

    # Create directory structure
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "expected_output")

    os.makedirs(input_dir, exist_ok=True)

    print(f"Generating test images in {base_dir}/")
    print("=" * 50)

    for name, pattern in TEST_PATTERNS.items():
        print(f"\nGenerating: {name}")
        print(f"  Pattern: {pattern}")

        # Create input image
        pixels = create_cube_face_image(pattern)
        input_path = os.path.join(input_dir, f"{name}.ppm")
        write_ppm(input_path, 640, 480, pixels)
        print(f"  Input: {input_path}")

        # Create expected output directory
        pattern_output_dir = os.path.join(output_dir, name)
        os.makedirs(pattern_output_dir, exist_ok=True)

        # Create expected facelet images
        facelets = create_expected_facelets(pattern, size=64)
        for i, facelet_pixels in enumerate(facelets):
            facelet_path = os.path.join(pattern_output_dir, f"facelet_{i}.ppm")
            write_ppm(facelet_path, 64, 64, facelet_pixels)
        print(f"  Expected output: {pattern_output_dir}/facelet_0.ppm - facelet_8.ppm")

    # Create a metadata file
    metadata_path = os.path.join(base_dir, "test_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("Rubik's Cube Facelet Segmenter - Test Images\n")
        f.write("=" * 50 + "\n\n")
        f.write("Input images: 640x480 pixels, PPM format\n")
        f.write("Expected output: 9 x 64x64 pixel facelets per input\n\n")
        f.write("Facelet ordering:\n")
        f.write("  0 | 1 | 2\n")
        f.write("  ---------\n")
        f.write("  3 | 4 | 5\n")
        f.write("  ---------\n")
        f.write("  6 | 7 | 8\n\n")
        f.write("Color codes:\n")
        for code, rgb in COLORS.items():
            f.write(f"  {code}: RGB{rgb}\n")
        f.write("\nTest patterns:\n")
        for name, pattern in TEST_PATTERNS.items():
            f.write(f"  {name}: {pattern}\n")

    print(f"\n{'=' * 50}")
    print(f"Generated {len(TEST_PATTERNS)} test cases")
    print(f"Metadata: {metadata_path}")
    print("\nDirectory structure:")
    print(f"  {base_dir}/")
    print(f"    input/           # Input images (640x480)")
    print(f"    expected_output/ # Expected facelet outputs")
    print(f"      <test_name>/   # One folder per test")
    print(f"        facelet_0.ppm - facelet_8.ppm")


if __name__ == "__main__":
    generate_all_test_images()
    print("\nDone! To convert PPM to PNG (if needed):")
    print("  # On macOS/Linux with ImageMagick:")
    print("  convert input.ppm output.png")
    print("  # Or use: brew install imagemagick")
