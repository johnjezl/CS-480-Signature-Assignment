"""
Train Color Classifier CNN
Training script for Rubik's Cube color classification

Features:
- Train/Validation/Test split (70%/15%/15%)
- Data augmentation
- Training visualization
- Model checkpointing
- Early stopping
- Evaluation metrics

Usage:
    python tools/train_color_classifier.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import cv2
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import time

# Add parent directory to path so we can import from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from ColorClassifierCNN import ColorClassifierCNN


class FaceletDataset(Dataset):
    """
    Custom dataset for Rubik's Cube facelet images
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory containing color subdirectories
            transform: Optional transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform

        # Define color classes
        self.classes = ['white', 'yellow', 'red', 'orange', 'blue', 'green']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load all image paths and labels
        self.samples = []
        for color in self.classes:
            color_dir = os.path.join(root_dir, color)
            if os.path.exists(color_dir):
                for img_name in os.listdir(color_dir):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(color_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[color]))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image with OpenCV (BGR format)
        image = cv2.imread(img_path)

        # Convert BGR to RGB for PyTorch
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms():
    """
    Define data augmentation transforms for training and validation

    Returns:
        train_transform, val_transform
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),  # Slight rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def split_dataset(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets

    Args:
        dataset: PyTorch Dataset
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    print(f"\nDataset split:")
    print(f"  Training:   {len(train_dataset)} images ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_dataset)} images ({val_ratio*100:.0f}%)")
    print(f"  Test:       {len(test_dataset)} images ({test_ratio*100:.0f}%)")

    return train_dataset, val_dataset, test_dataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch

    Returns:
        Average loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                         'acc': f'{100*correct/total:.2f}%'})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validate model

    Returns:
        Average loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total

    return val_loss, val_acc


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model and generate detailed metrics

    Returns:
        accuracy, confusion matrix, classification report
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions,
                                   target_names=class_names, digits=4)

    return accuracy, cm, report


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """
    Plot training and validation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', marker='o')
    ax2.plot(val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def train_model(data_dir='datasets/training_dataset/synthetic',
                num_epochs=20,
                batch_size=32,
                learning_rate=0.001,
                device=None,
                save_dir='models'):
    """
    Main training function

    Args:
        data_dir: Directory containing training data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on (None for auto-detect)
        save_dir: Directory to save trained models
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Get transforms
    train_transform, val_transform = get_data_transforms()

    # Load full dataset with training transforms initially
    full_dataset = FaceletDataset(data_dir, transform=None)

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

    # Apply transforms to each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Create model
    model = ColorClassifierCNN(num_classes=6)
    model = model.to(device)

    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model size: {model.count_parameters() * 4 / 1024 / 1024:.2f} MB")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=3)

    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60 + "\n")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, model_path)
            print(f"  [SAVED] New best model! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    training_time = time.time() - start_time

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         save_path=os.path.join(save_dir, 'training_history.png'))

    # Load best model for final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)

    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    test_acc, cm, report = evaluate_model(model, test_loader, device,
                                         full_dataset.classes)

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    plot_confusion_matrix(cm, full_dataset.classes,
                         save_path=os.path.join(save_dir, 'confusion_matrix.png'))

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {training_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {os.path.join(save_dir, 'best_model.pth')}")
    print("="*60)

    return model, train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        data_dir='datasets/training_dataset/synthetic',
        num_epochs=20,
        batch_size=32,
        learning_rate=0.001,
        save_dir='models'
    )
