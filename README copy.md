# TestCube - Rubik's Cube Facelet Segmenter

A component that takes an image containing a Rubik's cube face and outputs 9 segmented 64x64 images of each facelet.

## Overview

This component processes a camera/image input (typically 640x480) containing a Rubik's cube face and:

1. Detects or accepts a bounding box for the cube face region
2. Extracts and squares the face region
3. Divides it into a 3x3 grid of 9 facelets
4. Outputs 9 oriented 64x64 images ready for the next processing stage

## Facelet Ordering

The output facelets are ordered top-left to bottom-right:

```
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from facelet_segmenter import FaceletSegmenter, BoundingBox

# Create segmenter (outputs 64x64 images by default)
segmenter = FaceletSegmenter(output_size=64)

# Option 1: Load from file with auto-detection
facelets = segmenter.segment_from_file("cube_face.jpg")

# Option 2: Load from file with explicit bounding box
bbox = BoundingBox(x=100, y=50, width=300, height=300)
facelets = segmenter.segment_from_file("cube_face.jpg", bbox=bbox)

# Option 3: Process numpy array directly
import cv2
image = cv2.imread("cube_face.jpg")
facelets = segmenter.segment(image, bbox=bbox)
```

### Functional Interface

```python
from facelet_segmenter import segment_cube_face

# Simple one-liner
facelets = segment_cube_face(image, output_size=64)
```

### Saving Results

```python
# Save all facelets to a directory
saved_paths = segmenter.save_facelets(facelets, "output/", prefix="facelet")
# Creates: output/facelet_0.png through output/facelet_8.png
```

### Output Format

- Each facelet is a numpy array of shape `(64, 64, 3)` in BGR format (OpenCV default)
- Returns a list of exactly 9 facelets
- Facelets are ordered left-to-right, top-to-bottom

## API Reference

### FaceletSegmenter

```python
class FaceletSegmenter:
    def __init__(self, output_size: int = 64)
    def segment(self, image: np.ndarray, bbox: Optional[BoundingBox] = None) -> List[np.ndarray]
    def segment_from_file(self, image_path: str, bbox: Optional[BoundingBox] = None) -> List[np.ndarray]
    def save_facelets(self, facelets: List[np.ndarray], output_dir: str, prefix: str = "facelet") -> List[str]
```

### BoundingBox

```python
@dataclass
class BoundingBox:
    x: int      # Left edge x coordinate
    y: int      # Top edge y coordinate
    width: int  # Width of the region
    height: int # Height of the region
```

## Testing

Run the test suite:

```bash
python test_segmenter.py
```

This creates a `test_output/` directory with:
- `original_synthetic.png` - A synthetic test image
- `test_facelet_0.png` through `test_facelet_8.png` - Individual facelets
- `facelets_visualization.png` - Visual grid of all 9 facelets

## Integration

This component is designed to fit into a processing pipeline:

```
Camera/Image Input (640x480)
         |
         v
  FaceletSegmenter
         |
         v
   9 x 64x64 images
         |
         v
  Color Recognition
  (next component)
```
