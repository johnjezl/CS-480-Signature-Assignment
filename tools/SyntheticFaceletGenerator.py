import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

class SyntheticFaceletGenerator:
    def __init__(self, palette='mixed'):
        """
        Initialize generator with color palette

        Args:
            palette: 'palette1', 'palette2', 'mixed', or 'standard'
        """
        # Define multiple color palettes for different cube brands
        # NOTE: OpenCV uses BGR format, not RGB!
        self.palettes = {
            'standard': {
                'white': (255, 255, 255),    # BGR: same as RGB
                'yellow': (0, 255, 255),     # BGR: (B, G, R)
                'red': (0, 0, 255),          # BGR: (B, G, R)
                'orange': (0, 165, 255),     # BGR: (B, G, R)
                'blue': (255, 0, 0),         # BGR: (B, G, R)
                'green': (0, 255, 0)         # BGR: same as RGB
            },
            'palette1': {
                'white': (255, 255, 255),
                'yellow': (0, 255, 255),
                'red': (0, 0, 255),
                'orange': (0, 100, 255),
                'blue': (187, 0, 0),
                'green': (0, 187, 0)
            },
            'palette2': {
                'white': (255, 255, 255),
                'yellow': (0, 213, 255),
                'red': (52, 18, 183),
                'orange': (0, 88, 255),
                'blue': (173, 70, 0),
                'green': (72, 155, 0)
            },
            'palette3': {
                'white': (255, 255, 255),
                'yellow': (15, 246, 242),
                'red': (15, 36, 246),
                'orange': (15, 137, 246),
                'blue': (246, 15, 39),
                'green': (15, 246, 29)
            },
            'palette4': {
                'white': (216, 217, 208),
                'yellow': (59, 239, 210),
                'red': (33, 28, 203),
                'orange': (52, 111, 237),
                'blue': (159, 98, 0),
                'green': (97, 229, 85)
            },
            'palette5': {
                'white': (221, 217, 221),
                'yellow': (33, 240, 239),
                'red': (55, 59, 238),
                'orange': (51, 142, 253),
                'blue': (190, 54, 33),
                'green': (83, 226, 94),
            }
        }

        self.palette_mode = palette
        self.colors = self.palettes.get(palette, self.palettes['standard'])
    
    def generate_facelet(self, color_name, size=64):
        """
        Generate a single synthetic facelet image

        Args:
            color_name: One of 'white', 'yellow', 'red', 'orange', 'blue', 'green'
            size: Size of the square facelet in pixels (default 64x64)

        Returns:
            numpy array of shape (size, size, 3) representing RGB image
        """
        # If in mixed mode, randomly select a palette for this facelet
        if self.palette_mode == 'mixed':
            palette = np.random.choice(['standard', 'palette1', 'palette2', 'palette3', 'palette4', 'palette5'])
            base_color = self.palettes[palette][color_name]
        else:
            base_color = self.colors[color_name]

        # Create base colored square
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = base_color

        # Add realistic variations:

        # Slight color variation (±10% RGB) - simulates lighting variations
        variation = np.random.uniform(-0.1, 0.1, size=(size, size, 3))
        img = img.astype(np.float32)
        img = img * (1 + variation)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Add texture/grain - simulates camera sensor noise
        noise = np.random.normal(0, 5, (size, size, 3))
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Add black borders (simulate grid lines between facelets)
        border_width = max(1, size // 32)  # Scale border with image size
        img[0:border_width, :] = [0, 0, 0]  # Top border
        img[-border_width:, :] = [0, 0, 0]  # Bottom border
        img[:, 0:border_width] = [0, 0, 0]  # Left border
        img[:, -border_width:] = [0, 0, 0]  # Right border

        return img

    def apply_lighting_variation(self, img):
        """
        Apply random brightness and shadow effects

        Args:
            img: Input image as numpy array

        Returns:
            Image with lighting variations applied
        """
        # Random overall brightness (0.6x to 1.4x)
        brightness_factor = np.random.uniform(0.6, 1.4)
        img = img.astype(np.float32)
        img = img * brightness_factor

        # Add shadow gradient (simulates directional lighting)
        if np.random.random() < 0.5:  # 50% chance of shadow
            height, width = img.shape[:2]
            # Create gradient from one corner
            y, x = np.ogrid[:height, :width]
            if np.random.random() < 0.5:
                gradient = 1 - (x / width) * 0.3  # Left to right shadow
            else:
                gradient = 1 - (y / height) * 0.3  # Top to bottom shadow
            gradient = gradient[:, :, np.newaxis]  # Add channel dimension
            img = img * gradient

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def apply_blur(self, img):
        """
        Apply random blur to simulate camera focus issues

        Args:
            img: Input image as numpy array

        Returns:
            Blurred image
        """
        blur_type = np.random.choice(['none', 'gaussian', 'motion'])

        if blur_type == 'gaussian':
            kernel_size = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif blur_type == 'motion':
            # Simple motion blur with random direction
            kernel_size = 5
            kernel = np.zeros((kernel_size, kernel_size))
            if np.random.random() < 0.5:
                kernel[kernel_size // 2, :] = 1  # Horizontal motion
            else:
                kernel[:, kernel_size // 2] = 1  # Vertical motion
            kernel = kernel / kernel_size
            img = cv2.filter2D(img, -1, kernel)

        return img

    def generate_augmented_facelet(self, color_name, size=64,
                                   apply_lighting=True, apply_blur_effect=True):
        """
        Generate a facelet with full augmentation pipeline

        Args:
            color_name: One of 'white', 'yellow', 'red', 'orange', 'blue', 'green'
            size: Size of the square facelet in pixels
            apply_lighting: Whether to apply lighting variations
            apply_blur_effect: Whether to apply blur

        Returns:
            Augmented facelet image
        """
        # Generate base facelet
        img = self.generate_facelet(color_name, size)

        # Apply optional augmentations
        if apply_lighting:
            img = self.apply_lighting_variation(img)

        if apply_blur_effect:
            img = self.apply_blur(img)

        return img

    def generate_dataset(self, output_dir='training_dataset/synthetic',
                        samples_per_color=850, size=64):
        """
        Generate complete synthetic dataset for all colors

        Args:
            output_dir: Directory to save generated images
            samples_per_color: Number of images to generate per color (default 850)
            size: Size of each facelet image

        Returns:
            Dictionary with statistics about generated dataset
        """
        import os

        stats = {
            'total_images': 0,
            'per_color': {}
        }

        for color in self.colors.keys():
            # Create directory for this color
            color_dir = os.path.join(output_dir, color)
            os.makedirs(color_dir, exist_ok=True)

            count = 0
            for i in range(samples_per_color):
                # Generate augmented facelet
                img = self.generate_augmented_facelet(color, size=size)

                # Save image
                filename = f'{i:04d}.png'
                filepath = os.path.join(color_dir, filename)
                cv2.imwrite(filepath, img)
                count += 1

            stats['per_color'][color] = count
            stats['total_images'] += count
            print(f"Generated {count} images for {color}")

        print(f"\nTotal images generated: {stats['total_images']}")
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Create generator instance with MIXED palette mode for robustness
    print("Creating generator with MIXED palette mode...")
    print("This will train the model to recognize colors from multiple cube brands!\n")
    generator = SyntheticFaceletGenerator(palette='mixed')

    # Test 1: Generate a single facelet
    print("Test 1: Generating single facelets...")
    test_img = generator.generate_facelet('red', size=64)
    cv2.imwrite('test_facelet_red.png', test_img)
    print("Saved test_facelet_red.png")

    # Test 2: Generate augmented facelet
    print("\nTest 2: Generating augmented facelet...")
    test_aug = generator.generate_augmented_facelet('blue', size=64)
    cv2.imwrite('test_facelet_blue_augmented.png', test_aug)
    print("Saved test_facelet_blue_augmented.png")

    # Test 3: Generate small test dataset (10 samples per color for quick test)
    print("\nTest 3: Generating small test dataset...")
    stats = generator.generate_dataset(
        output_dir='test_dataset',
        samples_per_color=10,
        size=64
    )
    print("\nDataset statistics:", stats)
