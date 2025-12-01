"""
Test trained ColorClassifierCNN on real facelet images

This script evaluates how well the model trained on synthetic data
generalizes to real-world Rubik's Cube facelet images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import os
from pathlib import Path
from ColorClassifierCNN import ColorClassifierCNN


def load_model(model_path='models/best_model.pth', device=None):
    """Load trained model from checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ColorClassifierCNN(num_classes=6)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Model trained to epoch {checkpoint['epoch']+1} with val_acc={checkpoint['val_acc']:.2f}%")

    return model, device


def get_transform():
    """Get the same transform used during validation"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(model, image_path, transform, device, classes):
    """Predict class for a single image and return prediction with confidence"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transform
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_value = confidence.item() * 100

    return predicted_class, confidence_value, probabilities[0].cpu().numpy()


def test_on_real_data(data_dir='dataset/real_facelets', model_path='models/best_model.pth'):
    """
    Test model on real facelet images

    Args:
        data_dir: Directory containing real facelet images organized by color
        model_path: Path to trained model checkpoint
    """
    # Define classes (same order as training)
    classes = ['white', 'yellow', 'red', 'orange', 'blue', 'green']

    # Load model
    model, device = load_model(model_path)
    transform = get_transform()

    print(f"\nUsing device: {device}")
    print(f"\nTesting on real data from: {data_dir}")
    print("=" * 60)

    # Track results
    total_correct = 0
    total_images = 0
    results_by_class = {c: {'correct': 0, 'total': 0, 'predictions': []} for c in classes}
    all_results = []

    # Process each color directory
    for true_color in classes:
        color_dir = os.path.join(data_dir, true_color)
        if not os.path.exists(color_dir):
            continue

        images = [f for f in os.listdir(color_dir) if f.endswith('.png')]

        for img_name in images:
            img_path = os.path.join(color_dir, img_name)
            predicted_color, confidence, probs = predict_single_image(
                model, img_path, transform, device, classes
            )

            if predicted_color is None:
                continue

            is_correct = predicted_color == true_color
            total_images += 1
            results_by_class[true_color]['total'] += 1

            if is_correct:
                total_correct += 1
                results_by_class[true_color]['correct'] += 1
                status = "[OK]"
            else:
                status = "[X] "

            results_by_class[true_color]['predictions'].append({
                'image': img_name,
                'predicted': predicted_color,
                'confidence': confidence,
                'correct': is_correct
            })

            all_results.append({
                'true': true_color,
                'predicted': predicted_color,
                'confidence': confidence,
                'correct': is_correct,
                'image': img_name
            })

            # Print individual result
            print(f"{status} {true_color:8s} -> {predicted_color:8s} ({confidence:5.1f}%) | {img_name}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if total_images > 0:
        overall_accuracy = 100 * total_correct / total_images
        print(f"\nOverall Accuracy: {total_correct}/{total_images} = {overall_accuracy:.1f}%")

        print("\nPer-class breakdown:")
        print("-" * 40)
        for color in classes:
            stats = results_by_class[color]
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                print(f"  {color:8s}: {stats['correct']}/{stats['total']} correct ({acc:.0f}%)")

        # Show misclassifications
        misclassified = [r for r in all_results if not r['correct']]
        if misclassified:
            print("\nMisclassifications:")
            print("-" * 40)
            for r in misclassified:
                print(f"  {r['image']}: {r['true']} predicted as {r['predicted']} ({r['confidence']:.1f}%)")
        else:
            print("\nNo misclassifications! Perfect accuracy on real data.")
    else:
        print("No images found to test!")

    print("=" * 60)

    return overall_accuracy if total_images > 0 else 0, all_results


def visualize_predictions(data_dir='dataset/real_facelets', model_path='models/best_model.pth',
                          output_path='real_data_predictions.png'):
    """Create a visualization grid of predictions on real data"""
    import matplotlib.pyplot as plt

    classes = ['white', 'yellow', 'red', 'orange', 'blue', 'green']
    model, device = load_model(model_path)
    transform = get_transform()

    # Collect all images and predictions
    all_images = []
    for true_color in classes:
        color_dir = os.path.join(data_dir, true_color)
        if not os.path.exists(color_dir):
            continue

        for img_name in os.listdir(color_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(color_dir, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    predicted, confidence, _ = predict_single_image(
                        model, img_path, transform, device, classes
                    )
                    all_images.append({
                        'image': image_rgb,
                        'true': true_color,
                        'predicted': predicted,
                        'confidence': confidence
                    })

    if not all_images:
        print("No images to visualize!")
        return

    # Create grid
    n_images = len(all_images)
    cols = min(6, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3.5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for idx, data in enumerate(all_images):
        ax = axes[idx]
        ax.imshow(data['image'])

        is_correct = data['true'] == data['predicted']
        color = 'green' if is_correct else 'red'

        ax.set_title(f"True: {data['true']}\nPred: {data['predicted']} ({data['confidence']:.0f}%)",
                    fontsize=10, color=color)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(len(all_images), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Run test
    accuracy, results = test_on_real_data(
        data_dir='dataset/real_facelets',
        model_path='models/best_model.pth'
    )

    # Create visualization
    visualize_predictions(
        data_dir='dataset/real_facelets',
        model_path='models/best_model.pth',
        output_path='models/real_data_predictions.png'
    )
