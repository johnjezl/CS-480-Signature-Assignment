"""
RubiksCubeSolver - Main solver class that manages all components.

This class handles the lifecycle of expensive components (classifier, segmenters,
preprocessor) so they're instantiated once and reused across multiple operations.

Usage:
    solver = RubiksCubeSolver()

    # Process a single face
    colors, facelets = solver.process_face(image, face_name='front')

    # Process all 6 faces and solve
    solution = solver.solve_from_images(face_images)

    # Solve from camera (Jetson only)
    solution = solver.solve_from_camera(display=True)
"""

import os
import json
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from FaceletColorClassifier import FaceletColorClassifier
from ImagePreprocessor import ImagePreprocessor
from Segmenter import Segmenter
from adaptive_evaluator import AdaptiveEvaluator
from cube_evaluation import (
    evaluate_cube_result, FACE_NAMES, FACE_DISPLAY_NAMES,
    COLOR_TO_LETTER, EXPECTED_CENTERS
)
from IDASolver import KociembaSolver

# Try to import GPU preprocessor
try:
    from GPUImagePreprocessor import GPUImagePreprocessor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUImagePreprocessor = None

# Try to import Jetson camera
try:
    from JetsonCamera import JetsonCamera
    JETSON_AVAILABLE = True
except ImportError:
    JETSON_AVAILABLE = False
    JetsonCamera = None


class RubiksCubeSolver:
    """
    Main solver class that manages all cube scanning and solving components.

    Components are instantiated once and reused to avoid repeated initialization costs.
    """

    def __init__(self, use_gpu: bool = True, debug: bool = False):
        """
        Initialize the solver with all required components.

        Args:
            use_gpu: If True and available, use GPU-accelerated preprocessing
            debug: If True, print debug information
        """
        self.debug = debug
        self._debug_print("Initializing RubiksCubeSolver...")

        start_time = time.time()

        # Initialize preprocessor
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            try:
                self.preprocessor = GPUImagePreprocessor()
                self._debug_print("  Using GPU preprocessor")
            except Exception as e:
                self._debug_print(f"  GPU preprocessor failed: {e}, falling back to CPU")
                self.preprocessor = ImagePreprocessor()
                self.use_gpu = False
        else:
            self.preprocessor = ImagePreprocessor()
            self._debug_print("  Using CPU preprocessor")

        # Initialize classifier (this is the expensive one)
        self._debug_print("  Loading color classifier...")
        classifier_start = time.time()
        self.classifier = FaceletColorClassifier()
        self._debug_print(f"    Loaded in {time.time() - classifier_start:.2f}s")

        # Cache for segmenters (created on demand)
        self._segmenters: Dict[str, Any] = {}

        # Adaptive evaluator (reusable)
        self.adaptive_evaluator = AdaptiveEvaluator()

        # Store last solution for reference
        self.last_cube_data: Optional[Dict] = None
        self.last_solution: Optional[str] = None

        total_time = time.time() - start_time
        self._debug_print(f"  Initialization complete in {total_time:.2f}s")

    def _debug_print(self, msg: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(msg)

    def get_segmenter(self, name: str = 'contour-neighbor') -> Any:
        """
        Get a segmenter by name, creating it if necessary.

        Segmenters are cached to avoid repeated creation.

        Args:
            name: Segmenter name (brightness-otsu, contour-neighbor, etc.)

        Returns:
            Segmenter instance
        """
        if name not in self._segmenters:
            self._debug_print(f"  Creating segmenter: {name}")
            self._segmenters[name] = Segmenter.create(name, debug=self.debug)
        return self._segmenters[name]

    def get_available_segmenters(self) -> List[str]:
        """Get list of available segmenter names."""
        return Segmenter.get_available_segmenters()

    def get_available_preprocessors(self) -> List[str]:
        """Get list of available preprocessing methods."""
        return self.preprocessor.get_available_methods()

    def process_face(self, image: np.ndarray, face_name: str = 'unknown',
                     segmenter_name: str = 'contour-neighbor',
                     seg_preprocess: Optional[str] = None,
                     cc_preprocess: Optional[str] = None,
                     display: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process a single cube face image and classify colors.

        Args:
            image: BGR image of the cube face
            face_name: Name of the face (up, down, front, back, left, right)
            segmenter_name: Segmentation algorithm to use
            seg_preprocess: Preprocessing method for segmentation
            cc_preprocess: Preprocessing method for color classification
            display: If True, display intermediate results

        Returns:
            Tuple of (classifications, facelets) or (None, None) on error
            - classifications: 3x3 array of (color_name, confidence) tuples
            - facelets: 3x3x64x64x3 array of facelet images
        """
        segmenter = self.get_segmenter(segmenter_name)

        # Apply segmentation preprocessing if specified
        if seg_preprocess and seg_preprocess.lower() != 'none':
            processed_image = self.preprocessor.apply(seg_preprocess, image)
        else:
            processed_image = image

        # Segment the face
        facelets = segmenter.segment(processed_image)
        if facelets is None:
            return None, None

        # Apply CC preprocessing and classify each facelet
        classifications = np.empty((3, 3), dtype=object)
        for row in range(3):
            for col in range(3):
                facelet = facelets[row, col]

                # Apply CC preprocessing if specified
                if cc_preprocess and cc_preprocess.lower() != 'none':
                    facelet = self.preprocessor.apply(cc_preprocess, facelet)

                # Classify
                color, confidence = self.classifier.classify_facelet(facelet)
                classifications[row, col] = (color, confidence)

        return classifications, facelets

    def solve_from_images(self, face_images: Dict[str, np.ndarray],
                          segmenter_name: str = 'contour-neighbor',
                          seg_preprocess: Optional[str] = None,
                          cc_preprocess: Optional[str] = None,
                          force_centers: bool = False,
                          adaptive: bool = False) -> Optional[str]:
        """
        Process 6 face images and solve the cube.

        Args:
            face_images: Dict mapping face names to BGR images
            segmenter_name: Segmentation algorithm to use
            seg_preprocess: Preprocessing method for segmentation
            cc_preprocess: Preprocessing method for color classification
            force_centers: If True, force center colors to expected values
            adaptive: If True, use adaptive evaluation with confirmation

        Returns:
            Solution string (space-separated moves) or None on error
        """
        if adaptive:
            return self._solve_adaptive(face_images, force_centers)
        else:
            return self._solve_standard(face_images, segmenter_name,
                                       seg_preprocess, cc_preprocess, force_centers)

    def _solve_standard(self, face_images: Dict[str, np.ndarray],
                        segmenter_name: str,
                        seg_preprocess: Optional[str],
                        cc_preprocess: Optional[str],
                        force_centers: bool) -> Optional[str]:
        """Standard solving with specified preprocessing."""
        cube_data = {}

        for face_name in FACE_NAMES:
            if face_name not in face_images:
                print(f"Error: Missing image for {face_name}")
                return None

            image = face_images[face_name]
            classifications, _ = self.process_face(
                image, face_name, segmenter_name,
                seg_preprocess, cc_preprocess
            )

            if classifications is None:
                print(f"Error: Failed to process {face_name}")
                return None

            # Convert classifications to color array
            colors = []
            for row in range(3):
                for col in range(3):
                    color, _ = classifications[row, col]
                    colors.append(COLOR_TO_LETTER.get(color, '?'))
            cube_data[face_name] = colors

        # Validate and solve
        return self._validate_and_solve(cube_data, force_centers)

    def _solve_adaptive(self, face_images: Dict[str, np.ndarray],
                        force_centers: bool) -> Optional[str]:
        """Adaptive solving with two-result confirmation."""
        print("\nAdaptive evaluation: searching for confirmed result...", end='', flush=True)

        result = self.adaptive_evaluator.evaluate(
            face_images, self.classifier, self.preprocessor,
            max_attempts=100,
            combos_per_round=5,
            background_samples=10,
            force_centers=force_centers,
            record_metrics=True,
            verbose=True
        )

        if not result['is_valid']:
            if result.get('valid_results_count', 0) > 0:
                print(f"\nWARNING: Found {result['valid_results_count']} valid result(s) but could not confirm.")
            else:
                print("\nError: No valid result found.")
            return None

        cube_data = result['cube_data']
        return self._run_solver(cube_data)

    def _validate_and_solve(self, cube_data: Dict[str, List[str]],
                            force_centers: bool) -> Optional[str]:
        """Validate cube data and run the solver."""
        # Validate the cube configuration
        is_valid, message = evaluate_cube_result(cube_data, force_centers=force_centers)

        if not is_valid:
            print(f"Error: Invalid cube configuration - {message}")
            return None

        return self._run_solver(cube_data)

    def _run_solver(self, cube_data: Dict[str, List[str]]) -> Optional[str]:
        """Run the Kociemba solver on validated cube data."""
        self.last_cube_data = cube_data

        # Write input file (KociembaSolver reads from this)
        input_path = "AStar_in.json"
        output_path = "AStar_out.txt"

        with open(input_path, 'w') as f:
            json.dump({"cube": cube_data}, f, indent=2)

        # Remove old output
        if os.path.exists(output_path):
            os.remove(output_path)

        # Run solver
        self._debug_print("Running Kociemba solver...")
        start_time = time.time()

        try:
            solver = KociembaSolver()
            solver.RubikAStar()

            solve_time = time.time() - start_time

            # Read solution
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    solution = f.read().strip()

                if solution.startswith("ERROR"):
                    print(f"Solver error: {solution}")
                    return None

                self.last_solution = solution
                self._debug_print(f"Solved in {solve_time:.2f}s")
                return solution
            else:
                print("Error: Solution file not found")
                return None

        except Exception as e:
            print(f"Error running solver: {e}")
            return None

    def load_face_images(self, directory: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load face images from a directory.

        Expects files named: up.jpg, down.jpg, front.jpg, back.jpg, left.jpg, right.jpg
        (case-insensitive, supports .jpg, .jpeg, .png)

        Args:
            directory: Path to directory containing face images

        Returns:
            Dict mapping face names to BGR images, or None on error
        """
        if not os.path.isdir(directory):
            print(f"Error: '{directory}' is not a directory")
            return None

        face_images = {}
        extensions = ['.jpg', '.jpeg', '.png']

        for face_name in FACE_NAMES:
            found = False
            for ext in extensions:
                # Try different case combinations
                for name_variant in [face_name, face_name.upper(), face_name.capitalize()]:
                    path = os.path.join(directory, name_variant + ext)
                    if os.path.exists(path):
                        image = cv2.imread(path)
                        if image is not None:
                            face_images[face_name] = image
                            found = True
                            break
                if found:
                    break

            if not found:
                print(f"Error: Could not find image for '{face_name}'")
                return None

        return face_images

    @staticmethod
    def is_jetson_available() -> bool:
        """Check if Jetson camera is available."""
        return JETSON_AVAILABLE

    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU preprocessing is available."""
        return GPU_AVAILABLE
