"""
Generate Full Synthetic Dataset for Rubik's Cube Color Classification
Task B1.1: Training Data Collection

This script generates 5,100 synthetic facelet images (850 per color)
with mixed color palettes to support multiple cube brands.
"""

from SyntheticFaceletGenerator import SyntheticFaceletGenerator
import time

def main():
    print("="*60)
    print("FULL DATASET GENERATION - Task B1.1")
    print("="*60)
    print()
    print("Configuration:")
    print("  - Palette mode: MIXED (4 palettes for diverse cube brands)")
    print("  - Images per color: 850")
    print("  - Total images: 5,100")
    print("  - Image size: 64x64 pixels")
    print("  - Output: training_dataset/synthetic/")
    print()
    print("Augmentations applied:")
    print("  - Color variation (+/-10% RGB)")
    print("  - Texture/grain (Gaussian noise)")
    print("  - Black borders (grid lines)")
    print("  - Lighting variation (0.6x to 1.4x brightness)")
    print("  - Shadow gradients (directional lighting)")
    print("  - Blur effects (Gaussian and motion blur)")
    print()
    print("="*60)
    print()

    # Confirm before starting
    response = input("Generate full dataset? This will take 2-3 minutes (y/n): ")
    if response.lower() != 'y':
        print("Dataset generation cancelled.")
        return

    print("\nStarting generation...")
    start_time = time.time()

    # Create generator with mixed palette mode
    generator = SyntheticFaceletGenerator(palette='mixed')

    # Generate full dataset
    stats = generator.generate_dataset(
        output_dir='training_dataset/synthetic',
        samples_per_color=850,
        size=64
    )

    elapsed_time = time.time() - start_time

    # Print summary
    print()
    print("="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print()
    print(f"Total images: {stats['total_images']}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Average speed: {stats['total_images']/elapsed_time:.1f} images/second")
    print()
    print("Per-color breakdown:")
    for color, count in stats['per_color'].items():
        print(f"  {color:8s}: {count} images")
    print()
    print("Dataset ready for training!")
    print("Next step: Task B1.2 - Small CNN Training")
    print()

if __name__ == "__main__":
    main()
