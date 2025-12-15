"""
Segmenter Front-End

Provides a unified interface for accessing different Rubik's cube face
segmentation algorithms. Each segmenter uses a different approach to
detect and extract the 9 facelets from a cube face image.

Usage:
    from Segmenter import Segmenter

    # List available segmenters
    segmenters = Segmenter.get_available_segmenters()

    # Create a segmenter by name
    segmenter = Segmenter.create('contour-neighbor')

    # Segment an image
    facelets = segmenter.segment(image)

    # Get description
    desc = Segmenter.get_description('contour-neighbor')
"""

import numpy as np
from typing import Dict, List, Optional, Type, Any

# Import all segmenter implementations
from facelet_segmenter import FaceletSegmenter
from facelet_segmenter_v2 import FaceletSegmenterV2
from facelet_segmenter_v3 import FaceletSegmenterV3
from facelet_segmenter_v4 import FaceletSegmenterV4
from facelet_segmenter_v5 import FaceletSegmenterV5


# Registry of segmenters with meaningful names
_SEGMENTERS: Dict[str, Dict[str, Any]] = {
    'grid-division': {
        'class': FaceletSegmenter,
        'description': 'Basic grid division - detects cube boundary and divides into 3x3 grid',
        'details': 'Original v1 segmenter. Finds the cube region and subdivides it evenly. '
                   'Works well with centered, axis-aligned cubes.'
    },
    'contour-perspective': {
        'class': FaceletSegmenterV2,
        'description': 'Contour detection with perspective correction',
        'details': 'V2 segmenter. Uses contour-based quadrilateral detection and homography '
                   'transform for perspective correction. Good for tilted cubes and complex backgrounds.'
    },
    'contour-neighbor': {
        'class': FaceletSegmenterV3,
        'description': 'Contour-based facelet detection with neighbor validation',
        'details': 'V3 segmenter. Directly detects individual square facelets and validates them '
                   'by checking neighbor relationships. A valid facelet must have at least one neighbor.'
    },
    'canny-square': {
        'class': FaceletSegmenterV4,
        'description': 'Canny edge detection with square finding',
        'details': 'V4 segmenter. Uses Canny edge detection and contour analysis to find '
                   'square-shaped regions, then groups them into a 3x3 grid pattern.'
    },
    'brightness-otsu': {
        'class': FaceletSegmenterV5,
        'description': 'Brightness-based detection with Otsu thresholding',
        'details': 'V5 segmenter (Greg\'s CV approach). Uses HSV value channel with Otsu '
                   'thresholding to detect bright stickers against dark cube plastic. '
                   'Works well when stickers are brighter than the cube body.'
    },
}

# Default segmenter to use
DEFAULT_SEGMENTER = 'contour-neighbor'


class Segmenter:
    """
    Front-end class for Rubik's cube face segmentation.

    Provides a unified interface to access different segmentation algorithms
    by meaningful names rather than version numbers.
    """

    @staticmethod
    def get_available_segmenters() -> List[str]:
        """Get list of available segmenter names."""
        return sorted(_SEGMENTERS.keys())

    @staticmethod
    def get_description(name: str) -> str:
        """
        Get short description for a segmenter.

        Args:
            name: Segmenter name (case-insensitive)

        Returns:
            Short description string
        """
        name = name.lower()
        if name not in _SEGMENTERS:
            return f"Unknown segmenter: {name}"
        return _SEGMENTERS[name]['description']

    @staticmethod
    def get_details(name: str) -> str:
        """
        Get detailed description for a segmenter.

        Args:
            name: Segmenter name (case-insensitive)

        Returns:
            Detailed description string
        """
        name = name.lower()
        if name not in _SEGMENTERS:
            return f"Unknown segmenter: {name}"
        return _SEGMENTERS[name]['details']

    @staticmethod
    def get_all_descriptions() -> Dict[str, str]:
        """Get all segmenter names and their descriptions."""
        return {name: info['description'] for name, info in sorted(_SEGMENTERS.items())}

    @staticmethod
    def create(name: str = DEFAULT_SEGMENTER, output_size: int = 64, **kwargs) -> Any:
        """
        Create a segmenter instance by name.

        Args:
            name: Segmenter name (case-insensitive). Options:
                - 'grid-division': Basic grid subdivision (v1)
                - 'contour-perspective': Perspective correction (v2)
                - 'contour-neighbor': Neighbor-validated detection (v3)
                - 'canny-square': Canny + square detection (v4)
                - 'brightness-otsu': Otsu thresholding (v5)
            output_size: Size of output facelet images (default 64x64)
            **kwargs: Additional arguments passed to the segmenter constructor

        Returns:
            Segmenter instance

        Raises:
            ValueError: If segmenter name is not recognized
        """
        name = name.lower()
        if name not in _SEGMENTERS:
            available = ', '.join(sorted(_SEGMENTERS.keys()))
            raise ValueError(f"Unknown segmenter: '{name}'. Available: {available}")

        segmenter_class = _SEGMENTERS[name]['class']
        return segmenter_class(output_size=output_size, **kwargs)

    @staticmethod
    def get_class(name: str) -> Type:
        """
        Get the segmenter class by name.

        Args:
            name: Segmenter name (case-insensitive)

        Returns:
            Segmenter class

        Raises:
            ValueError: If segmenter name is not recognized
        """
        name = name.lower()
        if name not in _SEGMENTERS:
            available = ', '.join(sorted(_SEGMENTERS.keys()))
            raise ValueError(f"Unknown segmenter: '{name}'. Available: {available}")

        return _SEGMENTERS[name]['class']

    @staticmethod
    def get_default() -> str:
        """Get the default segmenter name."""
        return DEFAULT_SEGMENTER

    @staticmethod
    def is_valid(name: str) -> bool:
        """Check if a segmenter name is valid."""
        return name.lower() in _SEGMENTERS

    @staticmethod
    def print_help():
        """Print help information about available segmenters."""
        print("\nAvailable Segmenters:")
        print("-" * 70)
        for name in sorted(_SEGMENTERS.keys()):
            info = _SEGMENTERS[name]
            default_marker = " (default)" if name == DEFAULT_SEGMENTER else ""
            print(f"\n  {name}{default_marker}")
            print(f"    {info['description']}")
        print("-" * 70)


if __name__ == '__main__':
    # Demo/test
    print("Segmenter Front-End")
    print("=" * 70)

    print(f"\nAvailable segmenters: {Segmenter.get_available_segmenters()}")
    print(f"Default segmenter: {Segmenter.get_default()}")

    Segmenter.print_help()

    # Test creating each segmenter
    print("\nTesting segmenter creation:")
    for name in Segmenter.get_available_segmenters():
        try:
            seg = Segmenter.create(name)
            print(f"  {name}: OK - {type(seg).__name__}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Test with a dummy image
    print("\nTesting segmentation with dummy image:")
    test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    seg = Segmenter.create('grid-division')
    result = seg.segment(test_img)
    print(f"  Input shape: {test_img.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Expected: (3, 3, 64, 64, 3)")
