"""
Facelet Color Classifier Wrapper

A wrapper class for ColorClassifierCNN that provides a simple interface
for classifying Rubik's Cube facelet colors.

NOTE: All image inputs use BGR format (OpenCV's native format).

Performance optimizations:
- Direct numpy->tensor conversion (no PIL)
- Batch inference for multiple facelets/faces
- Optional ONNX Runtime support for faster CPU inference
"""

import torch
import numpy as np
import cv2
import os
from ColorClassifierCNN import ColorClassifierCNN

# Try to import ONNX Runtime for optimized inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class FaceletColorClassifier:
    """
    Facelet color classifier class that wraps the CNN model
    for easy inference on facelet images.

    Optimized for batch inference and minimal preprocessing overhead.
    """

    # Pre-computed normalization constants (ImageNet values)
    NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path='models/best_model.pth', device=None, use_onnx=True):
        """
        Initialize the classifier with a trained model.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (None for auto-detect)
            use_onnx: If True and ONNX Runtime available, use it for faster CPU inference
        """
        # Define color classes (same order as training)
        self.classes = ['white', 'yellow', 'red', 'orange', 'blue', 'green']

        # Check for ONNX model
        onnx_path = model_path.replace('.pth', '.onnx')
        self.use_onnx = use_onnx and ONNX_AVAILABLE and os.path.exists(onnx_path)

        if self.use_onnx:
            # Use ONNX Runtime for inference
            self.ort_session = ort.InferenceSession(
                onnx_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.device = 'onnx'
            self.model = None
        else:
            # Use PyTorch for inference
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device

            # Load model
            self.model = ColorClassifierCNN(num_classes=6)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.ort_session = None

        # Pre-compute normalization tensors for faster processing
        self._norm_mean_tensor = None
        self._norm_std_tensor = None

    def _preprocess_facelet_fast(self, facelet):
        """
        Fast preprocessing without PIL conversion.

        Args:
            facelet: BGR image (64, 64, 3) uint8

        Returns:
            Normalized tensor (3, 64, 64) float32
        """
        # Ensure uint8
        if facelet.dtype != np.uint8:
            facelet = np.clip(facelet, 0, 255).astype(np.uint8)

        # BGR to RGB
        rgb = facelet[:, :, ::-1]

        # Convert to float32 and normalize to [0, 1]
        tensor = rgb.astype(np.float32) / 255.0

        # Transpose from HWC to CHW
        tensor = tensor.transpose(2, 0, 1)

        # Normalize with ImageNet stats
        tensor[0] = (tensor[0] - self.NORM_MEAN[0]) / self.NORM_STD[0]
        tensor[1] = (tensor[1] - self.NORM_MEAN[1]) / self.NORM_STD[1]
        tensor[2] = (tensor[2] - self.NORM_MEAN[2]) / self.NORM_STD[2]

        return tensor

    def _preprocess_batch_fast(self, facelets):
        """
        Fast batch preprocessing without PIL conversion.

        Args:
            facelets: List of BGR images (64, 64, 3) uint8

        Returns:
            Batch tensor (N, 3, 64, 64) float32
        """
        batch_size = len(facelets)
        batch = np.empty((batch_size, 3, 64, 64), dtype=np.float32)

        for i, facelet in enumerate(facelets):
            batch[i] = self._preprocess_facelet_fast(facelet)

        return batch

    def _run_inference(self, batch):
        """
        Run inference on a batch of preprocessed tensors.

        Args:
            batch: numpy array (N, 3, 64, 64) float32

        Returns:
            tuple: (predictions, confidences) as numpy arrays
        """
        if self.use_onnx:
            # ONNX Runtime inference
            outputs = self.ort_session.run(None, {self.input_name: batch})[0]
            # Softmax
            exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
            predictions = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)
        else:
            # PyTorch inference
            tensor = torch.from_numpy(batch).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
                predictions = predictions.cpu().numpy()
                confidences = confidences.cpu().numpy()

        return predictions, confidences

    def classify_facelet(self, facelet):
        """
        Classify color from the image provided.

        Args:
            facelet: A 64x64 pixel image to classify.
                     Each pixel is a BGR value (3 integers).
                     np.ndarray: (64, 64, 3) [height][width][BGR]

        Returns:
            classification: A color classification with confidence level.
                           tuple: (color, confidence)
                           - color: str, one of 'white', 'yellow', 'red', 'orange', 'blue', 'green'
                           - confidence: float, confidence percentage (0-100)
        """
        # Validate input
        if not isinstance(facelet, np.ndarray):
            raise TypeError("facelet must be a numpy array")
        if facelet.shape != (64, 64, 3):
            raise ValueError(f"facelet must have shape (64, 64, 3), got {facelet.shape}")

        # Preprocess
        tensor = self._preprocess_facelet_fast(facelet)
        batch = tensor[np.newaxis, ...]  # Add batch dimension

        # Run inference
        predictions, confidences = self._run_inference(batch)

        # Get results
        color = self.classes[predictions[0]]
        confidence_value = confidences[0] * 100

        return (color, confidence_value)

    def classify_face(self, face):
        """
        Classify all facelets for a cube face using batch inference.

        Args:
            face: A 3x3 matrix of 64x64 pixel images to classify.
                  Each pixel is a BGR value (3 integers).
                  np.ndarray: (3, 3, 64, 64, 3) [row][col][height][width][BGR]

        Returns:
            classifications: A 3x3 matrix of color classifications with confidence level.
                            np.ndarray: (3, 3) of tuple: (color, confidence)
        """
        # Validate input
        if not isinstance(face, np.ndarray):
            raise TypeError("face must be a numpy array")
        if face.shape != (3, 3, 64, 64, 3):
            raise ValueError(f"face must have shape (3, 3, 64, 64, 3), got {face.shape}")

        # Flatten to list of facelets
        facelets = [face[row, col] for row in range(3) for col in range(3)]

        # Batch preprocess and inference
        batch = self._preprocess_batch_fast(facelets)
        predictions, confidences = self._run_inference(batch)

        # Create output array
        classifications = np.empty((3, 3), dtype=object)

        # Fill results
        idx = 0
        for row in range(3):
            for col in range(3):
                color = self.classes[predictions[idx]]
                confidence = confidences[idx] * 100
                classifications[row, col] = (color, confidence)
                idx += 1

        return classifications

    # Alias for backwards compatibility
    classify_face_batch = classify_face

    def classify_multiple_faces(self, faces_dict):
        """
        Classify all facelets for multiple cube faces in a single batch.

        This is the most efficient method for classifying all 6 faces at once,
        as it batches all 54 facelets into a single inference call.

        Args:
            faces_dict: Dict of face_name -> facelets array (3, 3, 64, 64, 3)

        Returns:
            Dict of face_name -> classifications array (3, 3) of (color, confidence)
        """
        # Collect all facelets with their face/position info
        all_facelets = []
        face_positions = []  # [(face_name, row, col), ...]

        for face_name, face in faces_dict.items():
            if face.shape != (3, 3, 64, 64, 3):
                raise ValueError(f"Face {face_name} has invalid shape {face.shape}")

            for row in range(3):
                for col in range(3):
                    all_facelets.append(face[row, col])
                    face_positions.append((face_name, row, col))

        if not all_facelets:
            return {}

        # Batch preprocess and inference
        batch = self._preprocess_batch_fast(all_facelets)
        predictions, confidences = self._run_inference(batch)

        # Create output structure
        results = {face_name: np.empty((3, 3), dtype=object) for face_name in faces_dict}

        # Fill results
        for idx, (face_name, row, col) in enumerate(face_positions):
            color = self.classes[predictions[idx]]
            confidence = confidences[idx] * 100
            results[face_name][row, col] = (color, confidence)

        return results

    def classify_facelets_batch(self, facelets_list):
        """
        Classify a batch of individual facelets.

        Args:
            facelets_list: List of facelet images (64, 64, 3) BGR

        Returns:
            List of (color, confidence) tuples
        """
        if not facelets_list:
            return []

        # Batch preprocess and inference
        batch = self._preprocess_batch_fast(facelets_list)
        predictions, confidences = self._run_inference(batch)

        # Return results
        return [(self.classes[pred], conf * 100)
                for pred, conf in zip(predictions, confidences)]

    def export_to_onnx(self, output_path=None):
        """
        Export the PyTorch model to ONNX format for faster inference.

        Args:
            output_path: Path for the ONNX model (default: same as pth but .onnx)

        Returns:
            Path to the exported ONNX model
        """
        if self.model is None:
            raise RuntimeError("Cannot export: model was loaded from ONNX")

        if output_path is None:
            output_path = 'models/best_model.onnx'

        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64).to(self.device)

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"Model exported to {output_path}")
        return output_path


if __name__ == "__main__":
    import time

    # Test the classifier
    print("Testing FaceletColorClassifier...")
    print("=" * 50)

    # Initialize classifier
    classifier = FaceletColorClassifier(model_path='models/best_model.pth')
    print(f"Classifier initialized on device: {classifier.device}")
    print(f"Using ONNX: {classifier.use_onnx}")
    print(f"Color classes: {classifier.classes}")

    # Test 1: Classify a single facelet
    print("\n--- Test 1: Single Facelet Classification ---")

    # Load a test image
    test_image_path = 'dataset/real_facelets/red/facelet_20251130_184307_768.png'
    try:
        img = cv2.imread(test_image_path)
        if img is not None:
            color, confidence = classifier.classify_facelet(img)
            print(f"Image: {test_image_path}")
            print(f"Classification: {color} ({confidence:.1f}%)")
        else:
            print(f"Could not load test image: {test_image_path}")
    except Exception as e:
        print(f"Error loading test image: {e}")

    # Test 2: Create a synthetic face and classify it
    print("\n--- Test 2: Full Face Classification (Batch) ---")

    # Create a dummy 3x3 face with random colors
    face = np.random.randint(0, 256, size=(3, 3, 64, 64, 3), dtype=np.uint8)

    # Set some specific colors for testing (BGR format)
    face[0, 0] = np.full((64, 64, 3), [0, 0, 255], dtype=np.uint8)  # Red (BGR)
    face[1, 1] = np.full((64, 64, 3), [255, 255, 255], dtype=np.uint8)  # White (center)
    face[2, 2] = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)  # Blue (BGR)

    classifications = classifier.classify_face(face)

    print("Face classifications (3x3 grid):")
    for row in range(3):
        row_str = ""
        for col in range(3):
            color, conf = classifications[row, col]
            row_str += f"{color:8s}({conf:5.1f}%) "
        print(f"  {row_str}")

    # Test 3: Performance benchmark
    print("\n--- Test 3: Performance Benchmark ---")

    # Create 6 faces (full cube)
    faces = {f'face_{i}': np.random.randint(0, 256, size=(3, 3, 64, 64, 3), dtype=np.uint8)
             for i in range(6)}

    # Benchmark single facelet inference
    single_times = []
    for _ in range(10):
        start = time.time()
        for face in faces.values():
            for row in range(3):
                for col in range(3):
                    classifier.classify_facelet(face[row, col])
        single_times.append(time.time() - start)

    # Benchmark batch inference
    batch_times = []
    for _ in range(10):
        start = time.time()
        classifier.classify_multiple_faces(faces)
        batch_times.append(time.time() - start)

    print(f"Single facelet inference (54 calls): {np.mean(single_times)*1000:.1f}ms avg")
    print(f"Batch inference (1 call):            {np.mean(batch_times)*1000:.1f}ms avg")
    print(f"Speedup: {np.mean(single_times)/np.mean(batch_times):.1f}x")

    print("\n" + "=" * 50)
    print("All tests completed!")
