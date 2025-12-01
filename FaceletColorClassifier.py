"""
Facelet Color Classifier Wrapper

A wrapper class for ColorClassifierCNN that provides a simple interface
for classifying Rubik's Cube facelet colors.

NOTE: All image inputs use BGR format (OpenCV's native format).
"""

import torch
import numpy as np
import cv2
from torchvision import transforms
from ColorClassifierCNN import ColorClassifierCNN


class FaceletColorClassifier:
    """
    Facelet color classifier class that wraps the CNN model
    for easy inference on facelet images.
    """

    def __init__(self, model_path='models/best_model.pth', device=None):
        """
        Initialize the classifier with a trained model.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (None for auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Define color classes (same order as training)
        self.classes = ['white', 'yellow', 'red', 'orange', 'blue', 'green']

        # Load model
        self.model = ColorClassifierCNN(num_classes=6)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define preprocessing transform (same as validation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

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

        # Ensure uint8 format
        if facelet.dtype != np.uint8:
            facelet = np.clip(facelet, 0, 255).astype(np.uint8)

        # Convert BGR to RGB for the model
        facelet_rgb = cv2.cvtColor(facelet, cv2.COLOR_BGR2RGB)

        # Apply transform and add batch dimension
        input_tensor = self.transform(facelet_rgb).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Get results
        color = self.classes[predicted.item()]
        confidence_value = confidence.item() * 100  # Convert to percentage

        return (color, confidence_value)

    def classify_face(self, face):
        """
        Classify all facelets for a cube face.

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

        # Create output array
        classifications = np.empty((3, 3), dtype=object)

        # Classify each facelet
        for row in range(3):
            for col in range(3):
                facelet = face[row, col]
                classifications[row, col] = self.classify_facelet(facelet)

        return classifications

    def classify_face_batch(self, face):
        """
        Classify all facelets for a cube face using batch inference (faster).

        Args:
            face: A 3x3 matrix of 64x64 pixel images to classify.
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

        # Prepare batch of 9 facelets
        batch_tensors = []
        for row in range(3):
            for col in range(3):
                facelet = face[row, col]
                if facelet.dtype != np.uint8:
                    facelet = np.clip(facelet, 0, 255).astype(np.uint8)
                # Convert BGR to RGB for the model
                facelet_rgb = cv2.cvtColor(facelet, cv2.COLOR_BGR2RGB)
                tensor = self.transform(facelet_rgb)
                batch_tensors.append(tensor)

        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)

        # Run batch inference
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

        # Create output array
        classifications = np.empty((3, 3), dtype=object)

        # Fill results
        idx = 0
        for row in range(3):
            for col in range(3):
                color = self.classes[predictions[idx].item()]
                confidence = confidences[idx].item() * 100
                classifications[row, col] = (color, confidence)
                idx += 1

        return classifications


if __name__ == "__main__":
    import cv2

    # Test the classifier
    print("Testing FaceletColorClassifier...")
    print("=" * 50)

    # Initialize classifier
    classifier = FaceletColorClassifier(model_path='models/best_model.pth')
    print(f"Classifier initialized on device: {classifier.device}")
    print(f"Color classes: {classifier.classes}")

    # Test 1: Classify a single facelet
    print("\n--- Test 1: Single Facelet Classification ---")

    # Load a test image
    test_image_path = 'dataset/real_facelets/red/facelet_20251130_184307_768.png'
    try:
        img = cv2.imread(test_image_path)  # OpenCV loads as BGR
        if img is not None:
            # No conversion needed - classifier accepts BGR directly
            color, confidence = classifier.classify_facelet(img)
            print(f"Image: {test_image_path}")
            print(f"Classification: {color} ({confidence:.1f}%)")
        else:
            print(f"Could not load test image: {test_image_path}")
    except Exception as e:
        print(f"Error loading test image: {e}")

    # Test 2: Create a synthetic face and classify it
    print("\n--- Test 2: Full Face Classification ---")

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

    # Test 3: Batch classification
    print("\n--- Test 3: Batch Face Classification ---")
    classifications_batch = classifier.classify_face_batch(face)

    print("Face classifications (batch mode):")
    for row in range(3):
        row_str = ""
        for col in range(3):
            color, conf = classifications_batch[row, col]
            row_str += f"{color:8s}({conf:5.1f}%) "
        print(f"  {row_str}")

    print("\n" + "=" * 50)
    print("All tests completed!")
