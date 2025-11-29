"""
Generate Synthetic Dataset for Rubik's Cube Color Classification

This script generates 5,100 synthetic facelet images (850 per color)
with mixed color palettes to support multiple cube brands.
"""

from SyntheticFaceletGenerator import SyntheticFaceletGenerator
import time

def main():
    print("="*60)
    print("DATASET GENERATION")
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

if __name__ == "__main__":
    main()
