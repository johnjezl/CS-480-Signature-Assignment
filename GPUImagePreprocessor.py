"""
GPU-Accelerated Image Preprocessor for Rubik's Cube Detection

Uses NVIDIA VPI (Vision Programming Interface) to accelerate image preprocessing
on Jetson devices. Falls back to CPU implementation for unsupported operations.

Usage:
    preprocessor = GPUImagePreprocessor()

    # Apply a single method (uses GPU acceleration where available)
    processed = preprocessor.apply("bilateral", image)

    # List available methods
    methods = preprocessor.get_available_methods()
"""

import cv2
import numpy as np
from typing import Dict, Callable, List, Optional

# Try to import VPI
try:
    import vpi
    VPI_AVAILABLE = True
except ImportError:
    VPI_AVAILABLE = False


class GPUImagePreprocessor:
    """
    GPU-accelerated image preprocessor using NVIDIA VPI.
    Falls back to CPU (OpenCV) for operations not supported by VPI.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the preprocessor.

        Args:
            use_gpu: If True, use GPU acceleration where available.
                     If False, use CPU-only implementations.
        """
        self._use_gpu = use_gpu and VPI_AVAILABLE
        self._methods: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self._descriptions: Dict[str, str] = {}
        self._gpu_accelerated: Dict[str, bool] = {}
        self._register_methods()

        if self._use_gpu:
            # Pre-create VPI backend context
            self._backend = vpi.Backend.CUDA
        else:
            self._backend = None

    def _register_methods(self):
        """Register all available preprocessing methods."""

        # None / Original - no preprocessing
        self._register("none", self._none, "No preprocessing (original image)", gpu=False)

        # Bilateral Filter variants - GPU accelerated
        self._register("bilateral", self._bilateral,
                      "Bilateral filter (d=9, sigma=75) - edge-preserving smoothing", gpu=True)
        self._register("bilateral-strong", self._bilateral_strong,
                      "Strong bilateral filter (d=15, sigma=100)", gpu=True)

        # CLAHE variants - partial GPU (histogram eq is GPU, color conversion is CPU)
        self._register("clahe-lab", self._clahe_lab,
                      "CLAHE on LAB L-channel - adaptive contrast enhancement", gpu=False)
        self._register("clahe-hsv", self._clahe_hsv,
                      "CLAHE on HSV V-channel", gpu=False)

        # Sharpening - GPU accelerated via gaussian
        self._register("unsharp", self._unsharp_mask,
                      "Unsharp mask sharpening (amount=1.5)", gpu=True)

        # Histogram equalization - GPU accelerated
        self._register("histeq", self._hist_eq_channels,
                      "Histogram equalization per BGR channel", gpu=True)
        self._register("histeq-v", self._hist_eq_v,
                      "Histogram equalization on V channel only", gpu=True)

        # Morphological - GPU accelerated
        self._register("morph-open", self._morph_opening,
                      "Morphological opening - removes small bright spots", gpu=True)
        self._register("morph-close", self._morph_closing,
                      "Morphological closing - fills small dark holes", gpu=True)

        # Color adjustments - CPU only (color space conversion)
        self._register("satboost", self._saturation_boost,
                      "Saturation boost (1.5x) - makes colors more vivid", gpu=False)
        self._register("satboost-mild", self._saturation_boost_mild,
                      "Mild saturation boost (1.3x)", gpu=False)
        self._register("white-balance", self._white_balance,
                      "Gray world white balance correction", gpu=False)

        # Gamma correction - CPU (LUT-based, fast anyway)
        self._register("gamma-bright", self._gamma_brighten,
                      "Gamma correction to brighten (gamma=0.7)", gpu=False)
        self._register("gamma-dark", self._gamma_darken,
                      "Gamma correction to darken (gamma=1.5)", gpu=False)

        # Blur/Denoise - GPU accelerated
        self._register("median", self._median_blur,
                      "Median blur (k=5) - salt-and-pepper noise removal", gpu=False)  # VPI median has issues
        self._register("gaussian", self._gaussian_blur,
                      "Gaussian blur (k=5)", gpu=True)

        # Contrast - CPU
        self._register("contrast-stretch", self._contrast_stretch,
                      "Contrast stretch (2-98 percentile)", gpu=False)

        # Combined methods
        self._register("bilateral-clahe", self._bilateral_clahe,
                      "Bilateral + CLAHE combined", gpu=True)
        self._register("bilateral-sat", self._bilateral_saturation,
                      "Bilateral + saturation boost", gpu=True)
        self._register("clahe-sat", self._clahe_saturation,
                      "CLAHE + saturation boost", gpu=False)
        self._register("full-pipeline", self._full_pipeline,
                      "Full pipeline: bilateral + CLAHE + saturation", gpu=True)

        # HSV saturation thresholding methods
        self._register("sat-threshold", self._saturation_threshold,
                      "HSV saturation thresholding - enhances colorful regions", gpu=False)
        self._register("sat-threshold-strong", self._saturation_threshold_strong,
                      "Strong HSV saturation thresholding - aggressive color enhancement", gpu=False)

        # Color masking for Rubik's cube colors
        self._register("cube-color-mask", self._cube_color_mask,
                      "Mask and enhance standard Rubik's cube colors (W,Y,R,O,B,G)", gpu=False)
        self._register("cube-color-mask-soft", self._cube_color_mask_soft,
                      "Soft color masking - gentler enhancement of cube colors", gpu=False)

    def _register(self, name: str, method: Callable[[np.ndarray], np.ndarray],
                  description: str, gpu: bool = False):
        """Register a preprocessing method."""
        self._methods[name.lower()] = method
        self._descriptions[name.lower()] = description
        self._gpu_accelerated[name.lower()] = gpu

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

    def is_gpu_accelerated(self, method_name: str) -> bool:
        """Check if a method uses GPU acceleration."""
        name = method_name.lower().replace('_', '-')
        return self._use_gpu and self._gpu_accelerated.get(name, False)

    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self._use_gpu

    # ========== VPI Helper Methods ==========

    def _vpi_process_channels(self, img: np.ndarray,
                               process_func: Callable) -> np.ndarray:
        """
        Process a BGR image by splitting channels and applying a VPI function to each.

        Args:
            img: Input BGR image
            process_func: Function that takes a vpi.Image and returns a vpi.Image

        Returns:
            Processed BGR image
        """
        if not self._use_gpu:
            raise RuntimeError("GPU not enabled")

        b, g, r = cv2.split(img)
        results = []

        with self._backend:
            for channel in [b, g, r]:
                vpi_img = vpi.asimage(channel, vpi.U8)
                vpi_out = process_func(vpi_img)

                with vpi_out.rlock_cpu() as data:
                    results.append(np.array(data))

        return cv2.merge(results)

    def _vpi_process_single(self, channel: np.ndarray,
                            process_func: Callable) -> np.ndarray:
        """
        Process a single-channel image with VPI.

        Args:
            channel: Input single-channel image
            process_func: Function that takes a vpi.Image and returns a vpi.Image

        Returns:
            Processed single-channel image
        """
        if not self._use_gpu:
            raise RuntimeError("GPU not enabled")

        with self._backend:
            vpi_img = vpi.asimage(channel, vpi.U8)
            vpi_out = process_func(vpi_img)

            with vpi_out.rlock_cpu() as data:
                return np.array(data)

    # ========== Preprocessing Methods ==========

    def _none(self, img: np.ndarray) -> np.ndarray:
        """No preprocessing - return copy."""
        return img.copy()

    def _bilateral(self, img: np.ndarray) -> np.ndarray:
        """Bilateral filter - edge-preserving smoothing."""
        if self._use_gpu:
            return self._vpi_process_channels(
                img, lambda x: x.bilateral_filter(9, 75, 75))
        else:
            return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def _bilateral_strong(self, img: np.ndarray) -> np.ndarray:
        """Strong bilateral filter."""
        if self._use_gpu:
            # VPI bilateral_filter max kernel size is 9, use higher sigma to compensate
            return self._vpi_process_channels(
                img, lambda x: x.bilateral_filter(9, 120, 120))
        else:
            return cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)

    def _clahe_lab(self, img: np.ndarray) -> np.ndarray:
        """CLAHE on LAB L-channel."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if self._use_gpu:
            l = self._vpi_process_single(l, lambda x: x.eqhist())
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _clahe_hsv(self, img: np.ndarray) -> np.ndarray:
        """CLAHE on HSV V-channel."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if self._use_gpu:
            v = self._vpi_process_single(v, lambda x: x.eqhist())
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)

        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _unsharp_mask(self, img: np.ndarray) -> np.ndarray:
        """Unsharp mask sharpening."""
        if self._use_gpu:
            gaussian = self._vpi_process_channels(
                img, lambda x: x.gaussian_filter(5, sigma=1.0))
        else:
            gaussian = cv2.GaussianBlur(img, (5, 5), 1.0)

        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    def _hist_eq_channels(self, img: np.ndarray) -> np.ndarray:
        """Histogram equalization per channel."""
        if self._use_gpu:
            return self._vpi_process_channels(img, lambda x: x.eqhist())
        else:
            channels = cv2.split(img)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            return cv2.merge(eq_channels)

    def _hist_eq_v(self, img: np.ndarray) -> np.ndarray:
        """Histogram equalization on V channel only."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if self._use_gpu:
            v = self._vpi_process_single(v, lambda x: x.eqhist())
        else:
            v = cv2.equalizeHist(v)

        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _morph_opening(self, img: np.ndarray) -> np.ndarray:
        """Morphological opening."""
        if self._use_gpu:
            kernel = (5, 5)  # VPI uses tuple for kernel size
            def morph_open(x):
                eroded = x.erode(kernel)
                return eroded.dilate(kernel)
            return self._vpi_process_channels(img, morph_open)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def _morph_closing(self, img: np.ndarray) -> np.ndarray:
        """Morphological closing."""
        if self._use_gpu:
            kernel = (5, 5)  # VPI uses tuple for kernel size
            def morph_close(x):
                dilated = x.dilate(kernel)
                return dilated.erode(kernel)
            return self._vpi_process_channels(img, morph_close)
        else:
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
        """Median blur - CPU only (VPI median has format issues)."""
        return cv2.medianBlur(img, 5)

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Gaussian blur."""
        if self._use_gpu:
            return self._vpi_process_channels(
                img, lambda x: x.gaussian_filter(5, sigma=1.0))
        else:
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

    def _saturation_threshold(self, img: np.ndarray) -> np.ndarray:
        """HSV saturation thresholding - enhances colorful regions."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        # Boost saturation for pixels that are already somewhat colorful
        mask = s > 30
        hsv[:, :, 1] = np.where(mask, np.clip(s * 1.5, 0, 255), s * 0.5)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _saturation_threshold_strong(self, img: np.ndarray) -> np.ndarray:
        """Strong HSV saturation thresholding - aggressive color enhancement."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        # More aggressive thresholding
        mask = s > 40
        hsv[:, :, 1] = np.where(mask, np.clip(s * 1.8, 0, 255), s * 0.3)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _cube_color_mask(self, img: np.ndarray) -> np.ndarray:
        """Mask and enhance standard Rubik's cube colors."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = img.copy().astype(np.float32)

        # Define HSV ranges for cube colors (H is 0-179 in OpenCV)
        color_ranges = [
            # White - low saturation, high value
            ((0, 0, 180), (179, 50, 255)),
            # Yellow - hue around 20-35
            ((18, 80, 150), (35, 255, 255)),
            # Red - hue around 0-10 or 170-179
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (179, 255, 255)),
            # Orange - hue around 10-22
            ((10, 100, 100), (22, 255, 255)),
            # Blue - hue around 100-130
            ((100, 100, 80), (130, 255, 255)),
            # Green - hue around 40-80
            ((40, 80, 80), (80, 255, 255)),
        ]

        # Create combined mask
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Enhance colors within the mask
        mask_3d = combined_mask[:, :, np.newaxis] / 255.0
        result = result * (1 + 0.3 * mask_3d)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _cube_color_mask_soft(self, img: np.ndarray) -> np.ndarray:
        """Soft color masking - gentler enhancement of cube colors."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = img.copy().astype(np.float32)

        # Wider HSV ranges for softer matching
        color_ranges = [
            # White - low saturation
            ((0, 0, 160), (179, 70, 255)),
            # Yellow
            ((15, 60, 130), (40, 255, 255)),
            # Red (both ends of hue spectrum)
            ((0, 70, 80), (12, 255, 255)),
            ((165, 70, 80), (179, 255, 255)),
            # Orange
            ((8, 80, 80), (25, 255, 255)),
            # Blue
            ((95, 70, 60), (135, 255, 255)),
            # Green
            ((35, 60, 60), (85, 255, 255)),
        ]

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Gentler enhancement
        mask_3d = combined_mask[:, :, np.newaxis] / 255.0
        result = result * (1 + 0.2 * mask_3d)
        return np.clip(result, 0, 255).astype(np.uint8)


# For backwards compatibility, create an alias
def get_preprocessor(use_gpu: bool = True) -> GPUImagePreprocessor:
    """
    Get a preprocessor instance.

    Args:
        use_gpu: If True, use GPU acceleration (default).
                 If False, use CPU-only implementations.

    Returns:
        GPUImagePreprocessor instance
    """
    return GPUImagePreprocessor(use_gpu=use_gpu)


def print_available_methods():
    """Print all available preprocessing methods and their descriptions."""
    preprocessor = GPUImagePreprocessor()
    print(f"\nGPU Acceleration: {'Enabled' if preprocessor.is_gpu_enabled() else 'Disabled'}")
    print(f"VPI Available: {VPI_AVAILABLE}")
    print("\nAvailable preprocessing methods:")
    print("-" * 70)
    for name in preprocessor.get_available_methods():
        desc = preprocessor.get_method_description(name)
        gpu_tag = " [GPU]" if preprocessor.is_gpu_accelerated(name) else ""
        print(f"  {name:<20} {desc}{gpu_tag}")
    print()


if __name__ == "__main__":
    print_available_methods()

    # Benchmark test
    import time

    print("\nRunning benchmark...")
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for use_gpu in [True, False]:
        if not VPI_AVAILABLE and use_gpu:
            continue

        preprocessor = GPUImagePreprocessor(use_gpu=use_gpu)
        mode = "GPU" if use_gpu else "CPU"

        print(f"\n{mode} Mode:")
        for method in ["bilateral", "gaussian", "histeq", "morph-open"]:
            times = []
            for _ in range(10):
                start = time.time()
                result = preprocessor.apply(method, test_img)
                times.append(time.time() - start)
            avg_time = np.mean(times[1:])  # Skip first (warmup)
            print(f"  {method:<20}: {avg_time*1000:.2f}ms")
