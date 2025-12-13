"""
RemoteDisplay - Stream OpenCV display windows to remote viewers

This module provides a drop-in replacement for cv2 display functions that also
streams the display to remote viewers via HTTP MJPEG streaming. Remote viewers
can connect from any device (Windows, Mac, Linux) using a web browser.

Usage:
    from RemoteDisplay import RemoteDisplay

    # Create remote display instance
    display = RemoteDisplay(port=8080, local_display=True)
    display.start()

    # Use like cv2
    display.imshow("Window Name", frame)
    key = display.waitKey(30)

    # Or use the global convenience functions
    from RemoteDisplay import remote_imshow, remote_waitKey
    remote_imshow("Window Name", frame)
    key = remote_waitKey(30)

    # Viewers connect to: http://<jetson-ip>:8080/
"""

import cv2
import numpy as np
import threading
import time
import socket
import struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import BytesIO
import json
import os
import sys
from typing import Dict, Optional, Tuple, Callable
from contextlib import contextmanager


# HTML template for the web viewer
VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Rubik's Cube Scanner - Remote Display</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 {
            font-size: 1.3em;
            font-weight: 500;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        .status-dot.disconnected {
            background: #ef4444;
            animation: none;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .container {
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .windows-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .window-card {
            background: #0f3460;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .window-header {
            background: #16213e;
            padding: 12px 15px;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .window-header .fps {
            font-size: 0.85em;
            color: #94a3b8;
        }
        .window-content {
            padding: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
            background: #000;
        }
        .window-content img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        .window-content .no-frame {
            color: #64748b;
            font-style: italic;
        }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            background: #e94560;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.2s;
        }
        button:hover {
            background: #ff6b6b;
        }
        button.secondary {
            background: #0f3460;
        }
        button.secondary:hover {
            background: #1a4a7a;
        }
        .info {
            margin-top: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
            font-size: 0.9em;
            color: #94a3b8;
        }
        .fullscreen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .fullscreen img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .fullscreen .exit-fullscreen {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            font-size: 1.5em;
        }
        @media (max-width: 600px) {
            .windows-grid {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 1.1em;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Rubik's Cube Scanner - Remote Display</h1>
        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Connected</span>
        </div>
    </div>
    <div class="container">
        <div class="windows-grid" id="windowsGrid">
            <!-- Window cards will be added dynamically -->
        </div>
        <div class="controls">
            <button onclick="refreshAll()">Refresh All</button>
            <button onclick="toggleAutoRefresh()" id="autoRefreshBtn">Pause Auto-Refresh</button>
            <button onclick="showAllWindows()" class="secondary">Show All Windows</button>
        </div>
        <div class="info">
            <p><strong>Connection:</strong> <span id="serverUrl"></span></p>
            <p><strong>Active Windows:</strong> <span id="windowCount">0</span></p>
            <p><strong>Tip:</strong> Click on any window to view fullscreen. Press Escape to exit.</p>
        </div>
    </div>
    <div class="fullscreen" id="fullscreenView" style="display: none;">
        <div class="exit-fullscreen" onclick="exitFullscreen()">&times;</div>
        <img id="fullscreenImg" src="">
    </div>

    <script>
        const windows = {};
        let autoRefresh = true;
        let connected = true;
        let fps = {};
        let lastFrameTime = {};

        // Update server URL display
        document.getElementById('serverUrl').textContent = window.location.href;

        // Poll for window list
        async function updateWindowList() {
            try {
                const response = await fetch('/windows');
                const windowList = await response.json();

                // Update connection status
                if (!connected) {
                    connected = true;
                    document.getElementById('statusDot').classList.remove('disconnected');
                    document.getElementById('statusText').textContent = 'Connected';
                }

                // Update window count
                document.getElementById('windowCount').textContent = windowList.length;

                // Add new windows, remove closed ones
                const grid = document.getElementById('windowsGrid');
                const existingWindows = new Set(Object.keys(windows));

                windowList.forEach(name => {
                    if (!windows[name]) {
                        // Create new window card
                        const card = createWindowCard(name);
                        grid.appendChild(card);
                        windows[name] = {
                            card: card,
                            img: card.querySelector('img'),
                            fpsSpan: card.querySelector('.fps')
                        };
                        fps[name] = 0;
                        lastFrameTime[name] = Date.now();
                    }
                    existingWindows.delete(name);
                });

                // Remove closed windows
                existingWindows.forEach(name => {
                    if (windows[name]) {
                        windows[name].card.remove();
                        delete windows[name];
                        delete fps[name];
                        delete lastFrameTime[name];
                    }
                });

            } catch (e) {
                if (connected) {
                    connected = false;
                    document.getElementById('statusDot').classList.add('disconnected');
                    document.getElementById('statusText').textContent = 'Disconnected';
                }
            }
        }

        function createWindowCard(name) {
            const card = document.createElement('div');
            card.className = 'window-card';
            card.innerHTML = `
                <div class="window-header">
                    <span>${escapeHtml(name)}</span>
                    <span class="fps">-- fps</span>
                </div>
                <div class="window-content" onclick="showFullscreen('${escapeHtml(name)}')">
                    <img src="/frame/${encodeURIComponent(name)}?t=${Date.now()}"
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"
                         onload="this.style.display='block'; this.nextElementSibling.style.display='none'; updateFps('${escapeHtml(name)}');">
                    <span class="no-frame" style="display: none;">No frame available</span>
                </div>
            `;
            return card;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function updateFps(name) {
            const now = Date.now();
            if (lastFrameTime[name]) {
                const delta = now - lastFrameTime[name];
                if (delta > 0) {
                    fps[name] = Math.round(1000 / delta);
                }
            }
            lastFrameTime[name] = now;

            if (windows[name] && windows[name].fpsSpan) {
                windows[name].fpsSpan.textContent = fps[name] + ' fps';
            }
        }

        function refreshFrame(name) {
            if (windows[name] && windows[name].img && autoRefresh) {
                windows[name].img.src = `/frame/${encodeURIComponent(name)}?t=${Date.now()}`;
            }
        }

        function refreshAll() {
            Object.keys(windows).forEach(refreshFrame);
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            document.getElementById('autoRefreshBtn').textContent =
                autoRefresh ? 'Pause Auto-Refresh' : 'Resume Auto-Refresh';
        }

        function showAllWindows() {
            updateWindowList();
            refreshAll();
        }

        function showFullscreen(name) {
            const view = document.getElementById('fullscreenView');
            const img = document.getElementById('fullscreenImg');
            img.src = `/frame/${encodeURIComponent(name)}?t=${Date.now()}`;
            view.style.display = 'flex';

            // Keep updating in fullscreen
            view.dataset.windowName = name;
            fullscreenRefresh();
        }

        function fullscreenRefresh() {
            const view = document.getElementById('fullscreenView');
            if (view.style.display === 'flex' && autoRefresh) {
                const name = view.dataset.windowName;
                const img = document.getElementById('fullscreenImg');
                img.src = `/frame/${encodeURIComponent(name)}?t=${Date.now()}`;
                setTimeout(fullscreenRefresh, 33); // ~30fps
            }
        }

        function exitFullscreen() {
            document.getElementById('fullscreenView').style.display = 'none';
        }

        // Escape key exits fullscreen
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                exitFullscreen();
            }
        });

        // Main update loop
        function mainLoop() {
            updateWindowList();
            if (autoRefresh) {
                refreshAll();
            }
            setTimeout(mainLoop, 100); // 10 fps for individual frame updates
        }

        // Start
        mainLoop();
    </script>
</body>
</html>
"""


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for streaming frames and serving the viewer."""

    # Reference to the RemoteDisplay instance (set by server)
    display_instance = None

    def log_message(self, format, *args):
        """Suppress default logging to avoid console spam."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            self._serve_viewer()
        elif path == '/windows':
            self._serve_window_list()
        elif path.startswith('/frame/'):
            window_name = path[7:]  # Remove '/frame/' prefix
            # URL decode the window name
            from urllib.parse import unquote
            window_name = unquote(window_name)
            self._serve_frame(window_name)
        elif path == '/stream':
            # Get window name from query parameter
            query = parse_qs(parsed.query)
            window_name = query.get('window', [None])[0]
            self._serve_mjpeg_stream(window_name)
        elif path == '/health':
            self._serve_health()
        else:
            self.send_error(404, 'Not Found')

    def _serve_viewer(self):
        """Serve the HTML viewer page."""
        content = VIEWER_HTML.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(content))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(content)

    def _serve_window_list(self):
        """Serve list of active windows as JSON."""
        if self.display_instance:
            windows = list(self.display_instance.get_window_names())
        else:
            windows = []

        content = json.dumps(windows).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(content)

    def _serve_frame(self, window_name: str):
        """Serve a single JPEG frame for a window."""
        if not self.display_instance:
            self.send_error(503, 'Display not initialized')
            return

        frame = self.display_instance.get_frame(window_name)
        if frame is None:
            # Send a placeholder image
            self.send_error(404, f'Window "{window_name}" not found')
            return

        # Encode as JPEG
        try:
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            content = jpeg.tobytes()

            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', len(content))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f'Error encoding frame: {e}')

    def _serve_mjpeg_stream(self, window_name: Optional[str]):
        """Serve continuous MJPEG stream."""
        if not self.display_instance:
            self.send_error(503, 'Display not initialized')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()

        try:
            while self.display_instance and self.display_instance.is_running():
                frame = self.display_instance.get_frame(window_name)
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')

                time.sleep(0.033)  # ~30 fps
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected

    def _serve_health(self):
        """Health check endpoint."""
        content = b'OK'
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)


class ThreadedHTTPServer(HTTPServer):
    """HTTP server that handles each request in a new thread."""

    allow_reuse_address = True
    daemon_threads = True

    def process_request(self, request, client_address):
        """Handle request in a new thread."""
        thread = threading.Thread(target=self.process_request_thread,
                                  args=(request, client_address))
        thread.daemon = True
        thread.start()

    def process_request_thread(self, request, client_address):
        """Process request and handle exceptions."""
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


class RemoteDisplay:
    """
    Drop-in replacement for cv2 display functions with remote streaming support.

    This class wraps OpenCV's display functions and simultaneously streams the
    displayed content to remote viewers via HTTP.

    Example:
        display = RemoteDisplay(port=8080, local_display=True)
        display.start()

        # Use just like cv2
        display.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)
        display.imshow("Preview", frame)
        key = display.waitKey(30)

        display.stop()
    """

    def __init__(self, port: int = 8080, local_display: bool = True,
                 auto_start: bool = False, verbose: bool = True):
        """
        Initialize the remote display.

        Args:
            port: HTTP server port for remote viewers
            local_display: If True, also show windows locally via cv2
            auto_start: If True, start the server immediately
            verbose: If True, print status messages
        """
        self.port = port
        self.local_display = local_display
        self.verbose = verbose

        self._frames: Dict[str, np.ndarray] = {}  # window_name -> frame
        self._frame_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._windows: Dict[str, dict] = {}  # window_name -> window info

        self._server: Optional[ThreadedHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # Composite frame for streaming all windows together
        self._composite_frame: Optional[np.ndarray] = None
        self._composite_lock = threading.Lock()

        if auto_start:
            self.start()

    def start(self) -> bool:
        """
        Start the HTTP streaming server.

        Returns:
            bool: True if server started successfully
        """
        if self._running:
            return True

        try:
            # Configure handler with reference to this instance
            StreamingHandler.display_instance = self

            self._server = ThreadedHTTPServer(('0.0.0.0', self.port), StreamingHandler)
            self._server_thread = threading.Thread(target=self._server.serve_forever)
            self._server_thread.daemon = True
            self._server_thread.start()

            self._running = True

            if self.verbose:
                # Get local IP address
                local_ip = self._get_local_ip()
                print(f"\n{'='*60}")
                print(f"  Remote Display Server Started")
                print(f"{'='*60}")
                print(f"  Local URL:  http://localhost:{self.port}/")
                print(f"  Network:    http://{local_ip}:{self.port}/")
                print(f"{'='*60}")
                print(f"  Open the URL in a browser on any device to view the display")
                print(f"{'='*60}\n")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error starting remote display server: {e}")
            return False

    def stop(self):
        """Stop the HTTP streaming server."""
        self._running = False

        if self._server:
            self._server.shutdown()
            self._server = None

        if self._server_thread:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        StreamingHandler.display_instance = None

        if self.verbose:
            print("Remote display server stopped")

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running

    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _ensure_lock(self, window_name: str):
        """Ensure a lock exists for the given window."""
        with self._global_lock:
            if window_name not in self._frame_locks:
                self._frame_locks[window_name] = threading.Lock()

    def namedWindow(self, window_name: str, flags: int = cv2.WINDOW_AUTOSIZE):
        """
        Create a named window.

        Args:
            window_name: Name of the window
            flags: Window flags (cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL, etc.)
        """
        with self._global_lock:
            self._windows[window_name] = {'flags': flags}
            self._ensure_lock(window_name)

        if self.local_display:
            cv2.namedWindow(window_name, flags)

    def imshow(self, window_name: str, frame: np.ndarray):
        """
        Display an image in a window and stream it to remote viewers.

        Args:
            window_name: Name of the window
            frame: Image to display (BGR numpy array)
        """
        if frame is None:
            return

        # Make a copy to avoid issues with frame being modified
        frame_copy = frame.copy()

        # Store for remote streaming
        self._ensure_lock(window_name)
        with self._frame_locks[window_name]:
            self._frames[window_name] = frame_copy

        # Update window registry
        with self._global_lock:
            if window_name not in self._windows:
                self._windows[window_name] = {'flags': cv2.WINDOW_AUTOSIZE}

        # Update composite frame
        self._update_composite()

        # Show locally
        if self.local_display:
            cv2.imshow(window_name, frame)

    def waitKey(self, delay: int = 0) -> int:
        """
        Wait for a key press.

        Args:
            delay: Delay in milliseconds (0 = wait indefinitely)

        Returns:
            int: Key code of pressed key, or -1 if no key pressed
        """
        if self.local_display:
            return cv2.waitKey(delay)
        else:
            # If no local display, just sleep for the delay
            if delay > 0:
                time.sleep(delay / 1000.0)
            return -1

    def destroyWindow(self, window_name: str):
        """
        Destroy a window.

        Args:
            window_name: Name of the window to destroy
        """
        with self._global_lock:
            if window_name in self._windows:
                del self._windows[window_name]
            if window_name in self._frames:
                del self._frames[window_name]
            if window_name in self._frame_locks:
                del self._frame_locks[window_name]

        if self.local_display:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    def destroyAllWindows(self):
        """Destroy all windows."""
        with self._global_lock:
            window_names = list(self._windows.keys())

        for name in window_names:
            self.destroyWindow(name)

        if self.local_display:
            cv2.destroyAllWindows()

    def setWindowProperty(self, window_name: str, prop_id: int, prop_value: float):
        """Set window property."""
        if self.local_display:
            cv2.setWindowProperty(window_name, prop_id, prop_value)

    def moveWindow(self, window_name: str, x: int, y: int):
        """Move window to specified position."""
        if self.local_display:
            cv2.moveWindow(window_name, x, y)

    def resizeWindow(self, window_name: str, width: int, height: int):
        """Resize window."""
        if self.local_display:
            cv2.resizeWindow(window_name, width, height)

    def get_frame(self, window_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get the current frame for a window.

        Args:
            window_name: Name of window, or None for composite of all windows

        Returns:
            The frame as a numpy array, or None if not available
        """
        if window_name is None:
            # Return composite of all windows
            with self._composite_lock:
                return self._composite_frame.copy() if self._composite_frame is not None else None

        self._ensure_lock(window_name)
        with self._frame_locks[window_name]:
            frame = self._frames.get(window_name)
            return frame.copy() if frame is not None else None

    def get_window_names(self) -> list:
        """Get list of active window names."""
        with self._global_lock:
            return list(self._windows.keys())

    def _update_composite(self):
        """Update the composite frame showing all windows."""
        with self._global_lock:
            window_names = list(self._windows.keys())

        if not window_names:
            return

        frames = []
        for name in window_names:
            frame = self.get_frame(name)
            if frame is not None:
                frames.append((name, frame))

        if not frames:
            return

        # Create composite image
        if len(frames) == 1:
            composite = frames[0][1]
        else:
            # Arrange in grid
            n = len(frames)
            cols = min(3, n)
            rows = (n + cols - 1) // cols

            # Find max dimensions
            max_h = max(f[1].shape[0] for f in frames)
            max_w = max(f[1].shape[1] for f in frames)

            # Scale down if needed
            target_w = 640
            scale = min(1.0, target_w / max_w)
            cell_w = int(max_w * scale)
            cell_h = int(max_h * scale)

            composite = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

            for i, (name, frame) in enumerate(frames):
                row = i // cols
                col = i % cols

                # Resize frame
                resized = cv2.resize(frame, (cell_w, cell_h))

                # Add label
                cv2.putText(resized, name, (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Place in composite
                y1 = row * cell_h
                x1 = col * cell_w
                composite[y1:y1+cell_h, x1:x1+cell_w] = resized

        with self._composite_lock:
            self._composite_frame = composite

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Global instance for convenience functions
_global_display: Optional[RemoteDisplay] = None
_global_lock = threading.Lock()


def get_display(port: int = 8080, local_display: bool = True,
                auto_start: bool = True) -> RemoteDisplay:
    """
    Get or create the global RemoteDisplay instance.

    Args:
        port: HTTP server port
        local_display: If True, also show windows locally
        auto_start: If True, start server automatically

    Returns:
        The global RemoteDisplay instance
    """
    global _global_display

    with _global_lock:
        if _global_display is None:
            _global_display = RemoteDisplay(port=port, local_display=local_display,
                                           auto_start=auto_start)
        return _global_display


def remote_imshow(window_name: str, frame: np.ndarray,
                  port: int = 8080, local_display: bool = True):
    """
    Convenience function to display an image with remote streaming.

    Args:
        window_name: Name of the window
        frame: Image to display
        port: HTTP server port (only used on first call)
        local_display: If True, also show locally (only used on first call)
    """
    display = get_display(port=port, local_display=local_display)
    display.imshow(window_name, frame)


def remote_waitKey(delay: int = 0) -> int:
    """
    Convenience function for waitKey with remote display.

    Args:
        delay: Delay in milliseconds

    Returns:
        Key code of pressed key, or -1
    """
    display = get_display()
    return display.waitKey(delay)


def remote_destroyAllWindows():
    """Convenience function to destroy all windows."""
    global _global_display
    if _global_display:
        _global_display.destroyAllWindows()


def remote_destroyWindow(window_name: str):
    """Convenience function to destroy a specific window."""
    global _global_display
    if _global_display:
        _global_display.destroyWindow(window_name)


def stop_remote_display():
    """Stop the global remote display server."""
    global _global_display
    if _global_display:
        _global_display.stop()
        _global_display = None


# Wrapper class that can be used as a drop-in cv2 replacement
class cv2_remote:
    """
    Drop-in replacement module for cv2 display functions.

    Usage:
        from RemoteDisplay import cv2_remote as cv2

        cv2.imshow("Window", frame)
        cv2.waitKey(30)
    """

    # Pass through non-display functions to real cv2
    def __getattr__(self, name):
        return getattr(cv2, name)

    @staticmethod
    def namedWindow(window_name, flags=cv2.WINDOW_AUTOSIZE):
        get_display().namedWindow(window_name, flags)

    @staticmethod
    def imshow(window_name, frame):
        get_display().imshow(window_name, frame)

    @staticmethod
    def waitKey(delay=0):
        return get_display().waitKey(delay)

    @staticmethod
    def destroyWindow(window_name):
        get_display().destroyWindow(window_name)

    @staticmethod
    def destroyAllWindows():
        get_display().destroyAllWindows()

    @staticmethod
    def setWindowProperty(window_name, prop_id, prop_value):
        get_display().setWindowProperty(window_name, prop_id, prop_value)

    @staticmethod
    def moveWindow(window_name, x, y):
        get_display().moveWindow(window_name, x, y)

    @staticmethod
    def resizeWindow(window_name, width, height):
        get_display().resizeWindow(window_name, width, height)


# Export cv2_remote instance
cv2_remote = cv2_remote()


# Test code
if __name__ == "__main__":
    print("RemoteDisplay Test")
    print("=" * 50)

    # Create display
    display = RemoteDisplay(port=8080, local_display=True, verbose=True)
    display.start()

    print("\nCreating test windows...")
    print("Open the URL shown above in a browser to view remotely")
    print("Press 'q' to quit\n")

    # Create some test frames
    frame_count = 0
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    try:
        while True:
            # Create animated test frames
            for i, color in enumerate(colors[:3]):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)

                # Draw animated circle
                cx = int(160 + 100 * np.sin(frame_count / 30.0 + i * 2))
                cy = int(120 + 80 * np.cos(frame_count / 30.0 + i * 2))
                cv2.circle(frame, (cx, cy), 30, color, -1)

                # Add label
                cv2.putText(frame, f"Window {i+1}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                display.imshow(f"Test Window {i+1}", frame)

            frame_count += 1

            key = display.waitKey(33)  # ~30fps
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    print("\nCleaning up...")
    display.destroyAllWindows()
    display.stop()
    print("Done!")
