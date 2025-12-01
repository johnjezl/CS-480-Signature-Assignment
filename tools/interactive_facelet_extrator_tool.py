import cv2
import numpy as np
import os
from datetime import datetime

class FaceletExtractor:
    def __init__(self, output_dir="facelet_dataset"):
        self.output_dir = output_dir
        self.current_image = None
        self.window_name = "Click to Extract 64x64 Facelet"
        
        # Create output directories for each color
        self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'white']
        for color in self.colors:
            os.makedirs(f"{output_dir}/{color}", exist_ok=True)
    
    def extract_facelets(self, image_path):
        """Extract 64x64 patches from image with mouse clicks"""
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            print(f"Could not load {image_path}")
            return
        
        print("\nControls:")
        print("- Click center of facelet to extract 64x64 region")
        print("- Press 'r/o/y/g/b/w' to set color class")
        print("- Press 'q' to move to next image")
        print("- Press ESC to exit")
        
        self.current_color = 'red'  # Default
        print(f"Current color: {self.current_color}")
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            display = self.current_image.copy()
            cv2.putText(display, f"Color: {self.current_color}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Color selection
            if key == ord('r'): self.current_color = 'red'
            elif key == ord('o'): self.current_color = 'orange'
            elif key == ord('y'): self.current_color = 'yellow'
            elif key == ord('g'): self.current_color = 'green'
            elif key == ord('b'): self.current_color = 'blue'
            elif key == ord('w'): self.current_color = 'white'
            elif key == ord('q'): break
            elif key == 27: return False  # ESC
            
            if key in [ord(c) for c in 'roygbw']:
                print(f"Switched to color: {self.current_color}")
        
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Extract 64x64 region centered at click"""
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.current_image.shape[:2]
            
            # Calculate bounds (ensure we don't go out of image)
            x1 = max(0, x - 32)
            y1 = max(0, y - 32)
            x2 = min(w, x1 + 64)
            y2 = min(h, y1 + 64)
            
            # Adjust if near edge
            if x2 - x1 < 64: x1 = x2 - 64
            if y2 - y1 < 64: y1 = y2 - 64
            
            # Extract facelet
            facelet = self.current_image[y1:y2, x1:x2]
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.output_dir}/{self.current_color}/facelet_{timestamp}.png"
            cv2.imwrite(filename, facelet)
            
            # Show preview
            cv2.imshow("Extracted Facelet", cv2.resize(facelet, (256, 256)))
            print(f"Saved: {filename}")
            
            # Draw rectangle on main image
            cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Usage
extractor = FaceletExtractor("dataset/real_facelets")

# Process multiple images
image_urls = [
    "rubiks4.jpg"
]

for image_path in image_urls:
    if not extractor.extract_facelets(image_path):
        break

cv2.destroyAllWindows()
