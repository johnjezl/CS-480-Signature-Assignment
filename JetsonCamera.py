"""
JetsonCamera - Camera capture class for Jetson Orin Nano with IMX219-160 camera

This class provides an interface to capture images from the Waveshare IMX219-160
camera module connected to the Jetson Orin Nano via MIPI CSI.

Uses GStreamer pipeline through OpenCV for hardware-accelerated capture.
"""

import cv2
import numpy as np
import time
import platform
import os
import sys
import select


def is_display_available():
    """
    Check if a display is available for GUI operations.

    Returns:
        bool: True if display is available, False otherwise
    """
    # Check if DISPLAY environment variable is set (X11)
    display = os.environ.get('DISPLAY')
    if display:
        return True

    # Check for Wayland
    wayland = os.environ.get('WAYLAND_DISPLAY')
    if wayland:
        return True

    return False


def is_jetson():
    """
    Auto-detect if running on a Jetson device.

    Returns:
        bool: True if running on Jetson, False otherwise
    """
    # Check for Jetson-specific indicators

    # Method 1: Check for tegra in /proc/device-tree/model
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            if 'jetson' in model or 'tegra' in model:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    # Method 2: Check for NVIDIA in /etc/nv_tegra_release
    try:
        if os.path.exists('/etc/nv_tegra_release'):
            return True
    except:
        pass

    # Method 3: Check for nvarguscamerasrc availability
    try:
        result = os.popen('which nvarguscamerasrc 2>/dev/null').read()
        if result.strip():
            return True
    except:
        pass

    # Method 4: Check uname for tegra
    try:
        uname = platform.uname()
        if 'tegra' in uname.release.lower() or 'jetson' in uname.release.lower():
            return True
    except:
        pass

    return False


class JetsonCamera:
    """
    Camera interface for Jetson Orin Nano with IMX219-160 camera module.

    Uses GStreamer pipeline with nvarguscamerasrc for hardware-accelerated capture.
    Returns images in BGR format compatible with OpenCV and FaceletSegmenter.
    """

    def __init__(self, sensor_id=0, capture_width=1280, capture_height=720,
                 output_width=640, output_height=480, framerate=30, flip_method=0):
        """
        Initialize the Jetson camera.

        Args:
            sensor_id: CSI camera sensor ID (0 or 1)
            capture_width: Camera capture width
            capture_height: Camera capture height
            output_width: Output image width after scaling
            output_height: Output image height after scaling
            framerate: Capture framerate
            flip_method: Image flip method (0=none, 1=ccw90, 2=180, 3=cw90, etc.)
        """
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.output_width = output_width
        self.output_height = output_height
        self.framerate = framerate
        self.flip_method = flip_method
        self.cap = None

    def _build_gstreamer_pipeline(self):
        """
        Build the GStreamer pipeline string for nvarguscamerasrc.

        Returns:
            str: GStreamer pipeline string
        """
        # Pipeline for IMX219 camera on Jetson using nvarguscamerasrc
        # nvarguscamerasrc provides hardware-accelerated capture
        # nvvidconv handles colorspace conversion and scaling
        # Output is BGR format for OpenCV compatibility
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.capture_width}, "
            f"height=(int){self.capture_height}, format=(string)NV12, "
            f"framerate=(fraction){self.framerate}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.output_width}, "
            f"height=(int){self.output_height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink drop=1"
        )
        return pipeline

    def open(self):
        """
        Open the camera connection.

        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        if self.cap is not None and self.cap.isOpened():
            return True

        pipeline = self._build_gstreamer_pipeline()
        print(f"Opening camera with GStreamer pipeline...")

        try:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                print("Error: Failed to open camera with GStreamer pipeline")
                print("Make sure:")
                print("  1. IMX219 camera is properly connected to CSI port")
                print("  2. Camera is enabled in jetson-io.py")
                print("  3. OpenCV is built with GStreamer support")
                return False

            # Warm up - read a few frames to stabilize
            for _ in range(5):
                self.cap.read()

            print(f"Camera opened successfully ({self.output_width}x{self.output_height})")
            return True

        except Exception as e:
            print(f"Error opening camera: {e}")
            return False

    def close(self):
        """Close the camera connection."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera closed")

    def capture(self):
        """
        Capture a single frame from the camera.

        Returns:
            numpy.ndarray: BGR image (height, width, 3) or None on error
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                return None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            print("Error: Failed to capture frame")
            return None

        return frame

    def capture_with_preview(self, display=False):
        """
        Capture an image with live preview, waiting for user to press Enter.

        Args:
            display: If True, show live camera preview while waiting

        Returns:
            numpy.ndarray: BGR image (height, width, 3) or None on error
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                return None

        # Check if display is actually available
        if display and not is_display_available():
            print("Warning: --display requested but no display available (DISPLAY not set)")
            print("Continuing without visual preview...")
            display = False

        window_name = "Camera Preview"
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        captured_frame = None

        if display:
            # Display mode: show live preview, capture on Enter
            import select
            import sys

            print("Live preview active. Press Enter to capture (or 'q' + Enter to cancel)...")

            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Show "Press Enter to capture" on frame
                display_frame = frame.copy()
                text = "Press Enter to capture"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = (frame.shape[1] - text_width) // 2
                y = frame.shape[0] - 30

                cv2.putText(display_frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(display_frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)

                cv2.imshow(window_name, display_frame)
                cv2.waitKey(30)  # ~30fps, also processes window events

                # Check for terminal input (non-blocking)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == 'q':
                        cv2.destroyWindow(window_name)
                        print("Capture cancelled by user")
                        return None
                    else:
                        # Capture current frame
                        captured_frame = frame.copy()
                        display_frame = frame.copy()
                        cv2.putText(display_frame, "CAPTURED!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                        cv2.imshow(window_name, display_frame)
                        cv2.waitKey(500)
                        cv2.destroyWindow(window_name)
                        print("Captured!")
                        return captured_frame
        else:
            # No display mode: just prompt and capture
            print("Press Enter to capture (or 'q' to cancel)...")
            user_input = input("> ").strip().lower()

            if user_input == 'q':
                print("Capture cancelled by user")
                return None

            # Flush buffer and capture
            for _ in range(5):
                self.cap.read()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture frame")
                return None

            print("Captured!")
            return frame.copy()

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def display_image(image, window_name="Image", wait_key=True):
    """
    Display an image in a window.

    Args:
        image: BGR numpy array
        window_name: Name for the display window
        wait_key: If True, wait for Enter press before returning

    Returns:
        int: 0 if Enter pressed (if wait_key=True), -1 otherwise
    """
    if image is None:
        print("Error: No image to display")
        return -1

    if not is_display_available():
        print("Warning: Cannot display image - no display available (DISPLAY not set)")
        return -1

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, image)

    if wait_key:
        print("Press Enter to close the display...")
        while True:
            cv2.waitKey(30)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                break
        cv2.destroyWindow(window_name)
        return 0
    return -1


def display_images_grid(images, labels=None, window_name="Cube Faces", cols=3):
    """
    Display multiple images in a grid layout.

    Args:
        images: List of BGR numpy arrays
        labels: Optional list of labels for each image
        window_name: Name for the display window
        cols: Number of columns in the grid
    """
    if not images:
        print("Error: No images to display")
        return

    if not is_display_available():
        print("Warning: Cannot display images - no display available (DISPLAY not set)")
        return

    # Calculate grid dimensions
    n = len(images)
    rows = (n + cols - 1) // cols

    # Get max dimensions
    max_h = max(img.shape[0] for img in images if img is not None)
    max_w = max(img.shape[1] for img in images if img is not None)

    # Scale down if images are too large
    scale = 1.0
    target_width = 1920  # Max display width
    if max_w * cols > target_width:
        scale = target_width / (max_w * cols)
        max_w = int(max_w * scale)
        max_h = int(max_h * scale)

    # Create canvas
    canvas = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if img is None:
            continue

        row = i // cols
        col = i % cols

        # Resize image if needed
        if scale != 1.0:
            img = cv2.resize(img, (max_w, max_h))
        elif img.shape[0] != max_h or img.shape[1] != max_w:
            img = cv2.resize(img, (max_w, max_h))

        # Add label if provided
        if labels and i < len(labels):
            img = img.copy()
            cv2.putText(img, labels[i], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Place in canvas
        y1 = row * max_h
        y2 = y1 + max_h
        x1 = col * max_w
        x2 = x1 + max_w
        canvas[y1:y2, x1:x2] = img

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, canvas)

    print("Press Enter to close the display...")

    # Keep updating display while waiting for terminal input
    while True:
        cv2.waitKey(30)
        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
            break

    cv2.destroyWindow(window_name)


# Test code
if __name__ == "__main__":
    print("Jetson Camera Test")
    print("=" * 50)
    print(f"Jetson detected: {is_jetson()}")

    if is_jetson():
        print("\nTesting camera capture...")
        camera = JetsonCamera()

        if camera.open():
            # Test basic capture
            print("\nCapturing test frame...")
            frame = camera.capture()
            if frame is not None:
                print(f"Captured frame: {frame.shape}")
                display_image(frame, "Test Capture")

            # Test preview capture
            print("\nTesting preview capture...")
            frame = camera.capture_with_preview(display=True)
            if frame is not None:
                print(f"Captured frame: {frame.shape}")

            camera.close()
        else:
            print("Failed to open camera")
    else:
        print("\nNot running on Jetson - camera test skipped")
        print("This module requires a Jetson device with IMX219 camera")
