"""
Image Preprocessor for Rubik's Cube Detection

Provides various image preprocessing techniques that can be applied
before segmentation or color classification to improve accuracy.

Usage:
    preprocessor = ImagePreprocessor()

    # Apply a single method
    processed = preprocessor.apply("bilateral", image)

    # List available methods
    methods = preprocessor.get_available_methods()
"""

import cv2
import numpy as np
from typing import Dict, Callable, List, Optional


class ImagePreprocessor:
    """
    Image preprocessor with multiple techniques for improving
    cube detection and color classification.
    """

    def __init__(self):
        """Initialize the preprocessor with all available methods."""
        self._methods: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self._descriptions: Dict[str, str] = {}
        self._register_methods()

    def _register_methods(self):
        """Register all available preprocessing methods."""

        # None / Original - no preprocessing
        self._register("none", lambda img: img.copy(),
                      "No preprocessing (original image)")

        # Bilateral Filter variants
        self._register("bilateral", self._bilateral,
                      "Bilateral filter (d=9, sigma=75) - edge-preserving smoothing")
        self._register("bilateral-strong", self._bilateral_strong,
                      "Strong bilateral filter (d=15, sigma=100)")

        # CLAHE variants
        self._register("clahe-lab", self._clahe_lab,
                      "CLAHE on LAB L-channel - adaptive contrast enhancement")
        self._register("clahe-hsv", self._clahe_hsv,
                      "CLAHE on HSV V-channel")

        # Sharpening
        self._register("unsharp", self._unsharp_mask,
                      "Unsharp mask sharpening (amount=1.5)")

        # Histogram equalization
        self._register("histeq", self._hist_eq_channels,
                      "Histogram equalization per BGR channel")
        self._register("histeq-v", self._hist_eq_v,
                      "Histogram equalization on V channel only")

        # Morphological
        self._register("morph-open", self._morph_opening,
                      "Morphological opening - removes small bright spots")
        self._register("morph-close", self._morph_closing,
                      "Morphological closing - fills small dark holes")

        # Color adjustments
        self._register("satboost", self._saturation_boost,
                      "Saturation boost (1.5x) - makes colors more vivid")
        self._register("satboost-mild", self._saturation_boost_mild,
                      "Mild saturation boost (1.3x)")
        self._register("white-balance", self._white_balance,
                      "Gray world white balance correction")

        # Gamma correction
        self._register("gamma-bright", self._gamma_brighten,
                      "Gamma correction to brighten (gamma=0.7)")
        self._register("gamma-dark", self._gamma_darken,
                      "Gamma correction to darken (gamma=1.5)")

        # Blur/Denoise
        self._register("median", self._median_blur,
                      "Median blur (k=5) - salt-and-pepper noise removal")
        self._register("gaussian", self._gaussian_blur,
                      "Gaussian blur (k=5)")

        # Contrast
        self._register("contrast-stretch", self._contrast_stretch,
                      "Contrast stretch (2-98 percentile)")

        # Combined methods
        self._register("bilateral-clahe", self._bilateral_clahe,
                      "Bilateral + CLAHE combined")
        self._register("bilateral-sat", self._bilateral_saturation,
                      "Bilateral + saturation boost")
        self._register("clahe-sat", self._clahe_saturation,
                      "CLAHE + saturation boost")
        self._register("full-pipeline", self._full_pipeline,
                      "Full pipeline: bilateral + CLAHE + saturation")

    def _register(self, name: str, method: Callable[[np.ndarray], np.ndarray],
                  description: str):
        """Register a preprocessing method."""
        self._methods[name.lower()] = method
        self._descriptions[name.lower()] = description

    def apply(self, method_name: str, image: np.ndarray) -> np.ndarray:
        """
        Apply a preprocessing method to an image.

        Args:
            method_name: Name of the preprocessing method (case-insensitive)
            image: Input BGR image

        Returns:
            Preprocessed BGR image

        Raises:
            ValueError: If method_name is not recognized
        """
        name = method_name.lower().replace('_', '-')
        if name not in self._methods:
            available = ", ".join(sorted(self._methods.keys()))
            raise ValueError(f"Unknown preprocessing method: '{method_name}'. "
                           f"Available methods: {available}")

        return self._methods[name](image)

    def get_available_methods(self) -> List[str]:
        """Get list of available preprocessing method names."""
        return sorted(self._methods.keys())

    def get_method_description(self, method_name: str) -> str:
        """Get description for a preprocessing method."""
        name = method_name.lower()
        return self._descriptions.get(name, "No description available")

    def get_all_descriptions(self) -> Dict[str, str]:
        """Get all method names and their descriptions."""
        return dict(sorted(self._descriptions.items()))

    # ========== Individual Preprocessing Methods ==========

    def _bilateral(self, img: np.ndarray) -> np.ndarray:
        """Bilateral filter - edge-preserving smoothing."""
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def _bilateral_strong(self, img: np.ndarray) -> np.ndarray:
        """Strong bilateral filter."""
        return cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)

    def _clahe_lab(self, img: np.ndarray) -> np.ndarray:
        """CLAHE on LAB L-channel."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _clahe_hsv(self, img: np.ndarray) -> np.ndarray:
        """CLAHE on HSV V-channel."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _unsharp_mask(self, img: np.ndarray) -> np.ndarray:
        """Unsharp mask sharpening."""
        gaussian = cv2.GaussianBlur(img, (5, 5), 1.0)
        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    def _hist_eq_channels(self, img: np.ndarray) -> np.ndarray:
        """Histogram equalization per channel."""
        channels = cv2.split(img)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)

    def _hist_eq_v(self, img: np.ndarray) -> np.ndarray:
        """Histogram equalization on V channel only."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _morph_opening(self, img: np.ndarray) -> np.ndarray:
        """Morphological opening."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def _morph_closing(self, img: np.ndarray) -> np.ndarray:
        """Morphological closing."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def _saturation_boost(self, img: np.ndarray) -> np.ndarray:
        """Saturation boost (1.5x)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _saturation_boost_mild(self, img: np.ndarray) -> np.ndarray:
        """Mild saturation boost (1.3x)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _white_balance(self, img: np.ndarray) -> np.ndarray:
        """Gray world white balance."""
        result = img.copy().astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3

        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / max(avg_b, 1)), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / max(avg_g, 1)), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / max(avg_r, 1)), 0, 255)

        return result.astype(np.uint8)

    def _gamma_brighten(self, img: np.ndarray) -> np.ndarray:
        """Gamma correction to brighten."""
        gamma = 0.7
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(256)]).astype(np.uint8)
        return cv2.LUT(img, table)

    def _gamma_darken(self, img: np.ndarray) -> np.ndarray:
        """Gamma correction to darken."""
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(256)]).astype(np.uint8)
        return cv2.LUT(img, table)

    def _median_blur(self, img: np.ndarray) -> np.ndarray:
        """Median blur."""
        return cv2.medianBlur(img, 5)

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Gaussian blur."""
        return cv2.GaussianBlur(img, (5, 5), 0)

    def _contrast_stretch(self, img: np.ndarray) -> np.ndarray:
        """Contrast stretch using percentiles."""
        result = img.copy().astype(np.float32)
        for i in range(3):
            channel = result[:, :, i]
            min_val = np.percentile(channel, 2)
            max_val = np.percentile(channel, 98)
            if max_val > min_val:
                result[:, :, i] = np.clip(
                    (channel - min_val) * 255 / (max_val - min_val), 0, 255)
        return result.astype(np.uint8)

    def _bilateral_clahe(self, img: np.ndarray) -> np.ndarray:
        """Bilateral + CLAHE combined."""
        filtered = self._bilateral(img)
        return self._clahe_lab(filtered)

    def _bilateral_saturation(self, img: np.ndarray) -> np.ndarray:
        """Bilateral + saturation boost."""
        filtered = self._bilateral(img)
        return self._saturation_boost_mild(filtered)

    def _clahe_saturation(self, img: np.ndarray) -> np.ndarray:
        """CLAHE + saturation boost."""
        enhanced = self._clahe_lab(img)
        return self._saturation_boost_mild(enhanced)

    def _full_pipeline(self, img: np.ndarray) -> np.ndarray:
        """Full pipeline: bilateral + CLAHE + saturation."""
        filtered = self._bilateral(img)
        enhanced = self._clahe_lab(filtered)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def print_available_methods():
    """Print all available preprocessing methods and their descriptions."""
    preprocessor = ImagePreprocessor()
    print("\nAvailable preprocessing methods:")
    print("-" * 60)
    for name, desc in preprocessor.get_all_descriptions().items():
        print(f"  {name:<20} {desc}")
    print()


if __name__ == "__main__":
    print_available_methods()
