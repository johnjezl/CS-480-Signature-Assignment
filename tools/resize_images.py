import cv2
import os
from pathlib import Path

def resize_images(input_dir, output_dir, size=(640, 480)):
    """
    Resize all images in a directory to 640x480
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Read image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read {filename}")
                continue
            
            # Resize to 640x480
            resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            # Save resized image
            output_path = os.path.join(output_dir, f"resized_{filename}")
            cv2.imwrite(output_path, resized)
            
            print(f"Resized {filename}: {img.shape[:2]} -> {size}")
    
    print(f"Done! Resized images saved to {output_dir}")

# Usage
resize_images("input_faces/", "resized_640x480/")