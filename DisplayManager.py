"""
DisplayManager - Cross-platform display management for the Rubik's Cube Scanner

This module provides a unified display interface that works on:
- Jetson (with connected monitor)
- Mac (native OpenCV windows)
- Windows (native OpenCV windows)
- Optionally: Remote streaming to browsers

The goal is to abstract away platform differences so main.py can use
a single display interface that "just works" everywhere.

Usage:
    from DisplayManager import DisplayManager

    # Create manager - auto-detects platform
    dm = DisplayManager()

    # Simple display
    dm.imshow("Window Name", frame)
    key = dm.waitKey(30)

    # Higher-level functions
    dm.display_face_and_facelets(image, facelets, "Segmented: front")
    dm.display_images_grid(images, labels=["Up", "Down", ...])

    # Clean up
    dm.destroyAllWindows()
"""

import cv2
import numpy as np
import os
import sys
import time
import platform
import threading
from typing import Optional, Dict, List, Tuple, Callable
from contextlib import contextmanager


def get_platform() -> str:
    """
    Detect the current platform.

    Returns:
        'jetson', 'mac', 'windows', or 'linux'
    """
    system = platform.system().lower()

    if system == 'darwin':
        return 'mac'
    elif system == 'windows':
        return 'windows'
    elif system == 'linux':
        # Check if it's a Jetson
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'jetson' in model or 'tegra' in model:
                    return 'jetson'
        except (FileNotFoundError, PermissionError):
            pass

        # Check for tegra in kernel
        try:
            uname = platform.uname()
            if 'tegra' in uname.release.lower():
                return 'jetson'
        except:
            pass

        return 'linux'

    return 'unknown'


def is_display_available() -> bool:
    """
    Check if a display is available for GUI operations.

    Returns:
        bool: True if display is available
    """
    current_platform = get_platform()

    # Mac and Windows always have display capability
    if current_platform in ('mac', 'windows'):
        return True

    # Linux/Jetson - check for X11 or Wayland
    if os.environ.get('DISPLAY'):
        return True
    if os.environ.get('WAYLAND_DISPLAY'):
        return True

    return False


class DisplayManager:
    """
    Cross-platform display manager for OpenCV windows.

    This class provides a unified interface for displaying images that works
    across Jetson, Mac, and Windows platforms. It handles platform-specific
    differences internally.

    Features:
    - Standard cv2-like interface (imshow, waitKey, etc.)
    - Higher-level display functions (grids, face+facelets, etc.)
    - Optional remote streaming to browsers
    - Automatic platform detection
    """

    def __init__(self,
                 enable_remote: bool = False,
                 remote_port: int = 8080,
                 verbose: bool = False):
        """
        Initialize the display manager.

        Args:
            enable_remote: If True, also stream to remote browsers
            remote_port: Port for remote streaming server
            verbose: Print status messages
        """
        self.platform = get_platform()
        self.display_available = is_display_available()
        self.verbose = verbose
        self.enable_remote = enable_remote
        self.remote_port = remote_port

        self._windows: Dict[str, dict] = {}
        self._remote_display = None

        if self.verbose:
            print(f"DisplayManager initialized")
            print(f"  Platform: {self.platform}")
            print(f"  Display available: {self.display_available}")

        # Start remote streaming if requested
        if enable_remote:
            self._start_remote()

    def _start_remote(self):
        """Start the remote streaming server."""
        try:
            from RemoteDisplay import RemoteDisplay
            self._remote_display = RemoteDisplay(
                port=self.remote_port,
                local_display=False,  # We handle local display ourselves
                auto_start=True,
                verbose=self.verbose
            )
            if self.verbose:
                print(f"  Remote streaming: http://localhost:{self.remote_port}/")
        except ImportError:
            if self.verbose:
                print("  Remote streaming: Not available (RemoteDisplay not found)")

    def _check_display(self) -> bool:
        """Check if display is available, warn once if not."""
        if not self.display_available:
            if not hasattr(self, '_display_warned'):
                self._display_warned = True
                print("Warning: No display available. Display calls will be skipped.")
                print("  (Set DISPLAY environment variable or run on a machine with a monitor)")
            return False
        return True

    # =========================================================================
    # Core cv2-like interface
    # =========================================================================

    def namedWindow(self, window_name: str, flags: int = cv2.WINDOW_AUTOSIZE):
        """Create a named window."""
        self._windows[window_name] = {'flags': flags, 'created': True}

        if self._check_display():
            cv2.namedWindow(window_name, flags)

    def imshow(self, window_name: str, frame: np.ndarray):
        """
        Display an image in a window.

        Args:
            window_name: Name of the window
            frame: BGR image to display
        """
        if frame is None:
            return

        # Track window
        if window_name not in self._windows:
            self._windows[window_name] = {'flags': cv2.WINDOW_AUTOSIZE}

        # Stream to remote if enabled
        if self._remote_display:
            self._remote_display.imshow(window_name, frame)

        # Show locally
        if self._check_display():
            cv2.imshow(window_name, frame)

    def waitKey(self, delay: int = 0) -> int:
        """
        Wait for a key press.

        Args:
            delay: Delay in milliseconds (0 = wait indefinitely)

        Returns:
            Key code of pressed key, or -1 if no key pressed
        """
        if self.display_available:
            return cv2.waitKey(delay)
        else:
            # No display - just sleep
            if delay > 0:
                time.sleep(delay / 1000.0)
            return -1

    def destroyWindow(self, window_name: str):
        """Destroy a window."""
        if window_name in self._windows:
            del self._windows[window_name]

        if self._remote_display:
            self._remote_display.destroyWindow(window_name)

        if self.display_available:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    def destroyAllWindows(self):
        """Destroy all windows."""
        self._windows.clear()

        if self._remote_display:
            self._remote_display.destroyAllWindows()

        if self.display_available:
            cv2.destroyAllWindows()

    def setWindowProperty(self, window_name: str, prop_id: int, prop_value: float):
        """Set a window property."""
        if self.display_available:
            cv2.setWindowProperty(window_name, prop_id, prop_value)

    def moveWindow(self, window_name: str, x: int, y: int):
        """Move a window."""
        if self.display_available:
            cv2.moveWindow(window_name, x, y)

    def resizeWindow(self, window_name: str, width: int, height: int):
        """Resize a window."""
        if self.display_available:
            cv2.resizeWindow(window_name, width, height)

    # =========================================================================
    # Higher-level display functions (from JetsonCamera.py and main.py)
    # =========================================================================

    def display_image(self, image: np.ndarray,
                     window_name: str = "Image",
                     wait_key: bool = True) -> int:
        """
        Display an image in a window.

        Args:
            image: BGR numpy array
            window_name: Name for the display window
            wait_key: If True, wait for Enter press before returning

        Returns:
            0 if Enter pressed (if wait_key=True), -1 otherwise
        """
        if image is None:
            print("Error: No image to display")
            return -1

        if not self._check_display():
            return -1

        self.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        self.imshow(window_name, image)

        if wait_key:
            print("Press Enter to close the display...")
            self._wait_for_enter(window_name)
            self.destroyWindow(window_name)
            return 0

        return -1

    def display_images_grid(self, images: List[np.ndarray],
                           labels: Optional[List[str]] = None,
                           window_name: str = "Cube Faces",
                           cols: int = 3,
                           facelets_list: Optional[List] = None,
                           wait_key: bool = True):
        """
        Display multiple images in a grid layout.

        Args:
            images: List of BGR numpy arrays
            labels: Optional list of labels for each image
            window_name: Name for the display window
            cols: Number of columns in the grid
            facelets_list: Optional list of facelets arrays to overlay
            wait_key: If True, wait for Enter press
        """
        if not images:
            print("Error: No images to display")
            return

        if not self._check_display():
            return

        # Calculate grid dimensions
        n = len(images)
        rows = (n + cols - 1) // cols

        # Get max dimensions from valid images
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return

        max_h = max(img.shape[0] for img in valid_images)
        max_w = max(img.shape[1] for img in valid_images)

        # Calculate total canvas size before scaling
        total_width = max_w * cols
        total_height = max_h * rows

        # Get screen size for scaling, with safe fallback
        try:
            screen_width, screen_height = self.get_screen_size()
            if screen_width <= 0 or screen_height <= 0:
                screen_width, screen_height = 1920, 1080
        except:
            screen_width, screen_height = 1920, 1080

        # Leave some margin for window decorations and taskbar
        target_width = int(screen_width * 0.95)
        target_height = int(screen_height * 0.85)

        # Scale down if images are too large (consider both width AND height)
        scale = 1.0
        if total_width > target_width or total_height > target_height:
            scale_w = target_width / total_width if total_width > 0 else 1.0
            scale_h = target_height / total_height if total_height > 0 else 1.0
            scale = min(scale_w, scale_h)
            max_w = max(int(max_w * scale), 1)
            max_h = max(int(max_h * scale), 1)

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

            # Make a copy to draw on
            img = img.copy()

            # Add label if provided
            if labels and i < len(labels):
                cv2.putText(img, labels[i], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Overlay facelets if provided
            if facelets_list and i < len(facelets_list) and facelets_list[i] is not None:
                facelets = facelets_list[i]
                facelet_size = 32
                grid_width = 3 * facelet_size
                grid_height = 3 * facelet_size

                grid_x = (max_w - grid_width) // 2
                grid_y = max_h - grid_height - 5

                for r in range(3):
                    for c in range(3):
                        facelet = facelets[r, c]
                        facelet_resized = cv2.resize(facelet, (facelet_size, facelet_size))
                        fx = grid_x + c * facelet_size
                        fy = grid_y + r * facelet_size
                        img[fy:fy+facelet_size, fx:fx+facelet_size] = facelet_resized

                cv2.rectangle(img, (grid_x - 1, grid_y - 1),
                            (grid_x + grid_width, grid_y + grid_height),
                            (255, 255, 255), 1)

            # Place in canvas
            y1 = row * max_h
            x1 = col * max_w
            canvas[y1:y1+max_h, x1:x1+max_w] = img

        self.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        self.imshow(window_name, canvas)

        if wait_key:
            print("Press Enter to close the display...")
            self._wait_for_enter(window_name)
            self.destroyWindow(window_name)

    def display_face_and_facelets(self, image: np.ndarray,
                                  facelets: np.ndarray,
                                  window_name: str = "Face and Facelets",
                                  max_height: int = 400,
                                  max_width: int = 800):
        """
        Display the face image alongside a 3x3 grid of extracted facelets.

        Args:
            image: BGR numpy array of the full face image
            facelets: numpy array of shape (3, 3, 64, 64, 3)
            window_name: Name for the display window
            max_height: Maximum height for the display
            max_width: Maximum width for the display
        """
        if not self._check_display():
            return

        cell_size = 80
        border = 2
        spacing = 2

        bordered_size = cell_size + border * 2
        grid_size = bordered_size * 3 + spacing * 2

        # Black background for facelet grid
        facelet_grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        for row in range(3):
            for col in range(3):
                facelet = facelets[row, col]
                facelet_resized = cv2.resize(facelet, (cell_size, cell_size))
                bordered = cv2.copyMakeBorder(
                    facelet_resized, border, border, border, border,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                y1 = row * (bordered_size + spacing)
                x1 = col * (bordered_size + spacing)
                facelet_grid[y1:y1+bordered_size, x1:x1+bordered_size] = bordered

        # Scale face to match facelet grid height
        scale = grid_size / image.shape[0]
        new_width = int(image.shape[1] * scale)
        face_scaled = cv2.resize(image, (new_width, grid_size))

        # Calculate combined width
        gap = spacing
        combined_width = face_scaled.shape[1] + gap + grid_size

        # Scale down if too large
        final_scale = min(1.0, max_height / grid_size, max_width / combined_width)
        if final_scale < 1.0:
            new_grid_size = int(grid_size * final_scale)
            new_face_width = int(face_scaled.shape[1] * final_scale)
            new_face_height = int(face_scaled.shape[0] * final_scale)
            facelet_grid = cv2.resize(facelet_grid, (new_grid_size, new_grid_size))
            face_scaled = cv2.resize(face_scaled, (new_face_width, new_face_height))
            grid_size = new_grid_size

        # Combine face and facelet grid side by side
        gap_img = np.zeros((grid_size, spacing, 3), dtype=np.uint8)
        combined = np.hstack([face_scaled, gap_img, facelet_grid])

        self.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        self.imshow(window_name, combined)
        self.waitKey(100)

    # =========================================================================
    # Animation support
    # =========================================================================

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get the screen size for fullscreen rendering.

        Returns:
            (width, height) tuple
        """
        if self.platform == 'mac':
            # Mac - try to get screen size
            try:
                from AppKit import NSScreen
                frame = NSScreen.mainScreen().frame()
                return int(frame.size.width), int(frame.size.height)
            except ImportError:
                pass

        elif self.platform == 'windows':
            # Windows - use ctypes
            try:
                import ctypes
                user32 = ctypes.windll.user32
                return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            except:
                pass

        elif self.platform in ('jetson', 'linux'):
            # Linux/Jetson - try xrandr
            try:
                import subprocess
                result = subprocess.run(['xrandr'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if '*' in line:
                        parts = line.split()[0].split('x')
                        return int(parts[0]), int(parts[1])
            except:
                pass

        # Default fallback
        return 1920, 1080

    def create_fullscreen_window(self, window_name: str) -> Tuple[int, int]:
        """
        Create a fullscreen window.

        Args:
            window_name: Name for the window

        Returns:
            (width, height) of the screen
        """
        if not self._check_display():
            return 1920, 1080

        width, height = self.get_screen_size()

        # Create window with WINDOW_NORMAL flag to allow resizing/fullscreen
        self.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.moveWindow(window_name, 0, 0)
        self.resizeWindow(window_name, width, height)

        # Process window events to ensure window is visible
        self.waitKey(1)

        return width, height

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _wait_for_enter(self, window_name: str = None):
        """Wait for Enter key press while keeping window responsive."""
        import select

        while True:
            self.waitKey(30)

            # Check for terminal input (works on Unix-like systems)
            try:
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()
                    break
            except:
                # On Windows, select doesn't work with stdin
                # Fall back to simple waitKey
                key = self.waitKey(100)
                if key == 13 or key == 10:  # Enter key
                    break

    def get_window_names(self) -> List[str]:
        """Get list of active window names."""
        return list(self._windows.keys())

    def stop(self):
        """Clean up and stop the display manager."""
        self.destroyAllWindows()

        if self._remote_display:
            self._remote_display.stop()
            self._remote_display = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# =========================================================================
# Convenience functions for drop-in replacement
# =========================================================================

# Global instance
_global_manager: Optional[DisplayManager] = None


def get_manager(enable_remote: bool = False,
                remote_port: int = 8080,
                verbose: bool = False) -> DisplayManager:
    """
    Get or create the global DisplayManager instance.

    Args:
        enable_remote: Enable remote streaming
        remote_port: Port for remote streaming
        verbose: Print status messages

    Returns:
        The global DisplayManager instance
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = DisplayManager(
            enable_remote=enable_remote,
            remote_port=remote_port,
            verbose=verbose
        )
    return _global_manager


def display_image(image: np.ndarray,
                 window_name: str = "Image",
                 wait_key: bool = True) -> int:
    """Convenience function - display an image."""
    return get_manager().display_image(image, window_name, wait_key)


def display_images_grid(images: List[np.ndarray],
                       labels: Optional[List[str]] = None,
                       window_name: str = "Cube Faces",
                       cols: int = 3,
                       facelets_list: Optional[List] = None):
    """Convenience function - display images in a grid."""
    get_manager().display_images_grid(images, labels, window_name, cols, facelets_list)


# =========================================================================
# Test code
# =========================================================================

if __name__ == "__main__":
    print("DisplayManager Test")
    print("=" * 50)
    print(f"Platform: {get_platform()}")
    print(f"Display available: {is_display_available()}")
    print()

    if not is_display_available():
        print("No display available - skipping visual tests")
        sys.exit(0)

    dm = DisplayManager(verbose=True)

    print("\nTest 1: Basic imshow")
    print("-" * 30)

    # Create a test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "DisplayManager Test", (150, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(test_img, f"Platform: {dm.platform}", (200, 300),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
    cv2.putText(test_img, "Press any key to continue", (180, 400),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    dm.imshow("Test Window", test_img)
    dm.waitKey(0)
    dm.destroyWindow("Test Window")

    print("\nTest 2: Animation simulation")
    print("-" * 30)
    print("Press 'q' to stop")

    for i in range(100):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Animated circle
        cx = int(320 + 200 * np.sin(i / 15.0))
        cy = int(240 + 150 * np.cos(i / 15.0))
        cv2.circle(frame, (cx, cy), 40, (0, 255, 0), -1)

        cv2.putText(frame, f"Frame {i}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        dm.imshow("Animation Test", frame)
        key = dm.waitKey(33)
        if key == ord('q'):
            break

    dm.destroyAllWindows()
    print("\nTest complete!")
