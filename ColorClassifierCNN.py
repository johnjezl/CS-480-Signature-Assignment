"""
Color Classifier CNN Model Definition
Lightweight 3-layer CNN for Rubik's Cube Color Classification

Architecture:
- 3 Convolutional blocks
- Custom design optimized for 6-color classification
- Target: <5MB model, >95% accuracy, <10ms inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorClassifierCNN(nn.Module):
    """
    Lightweight 3-layer CNN for color classification

    Input: 64x64x3 RGB images
    Output: 6 classes (white, yellow, red, orange, blue, green)
    """

    def __init__(self, num_classes=6):
        super(ColorClassifierCNN, self).__init__()

        # Convolutional Block 1
        # Input: 64x64x3 -> Output: 32x32x16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32

        # Convolutional Block 2
        # Input: 32x32x16 -> Output: 16x16x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        # Convolutional Block 3
        # Input: 16x16x32 -> Output: 8x8x64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # Fully Connected Layers
        # After 3 pooling layers: 64 -> 32 -> 16 -> 8
        # Feature map size: 8x8x64 = 4096
        self.fc1 = nn.Linear(8 * 8 * 64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 8192)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """
    Calculate model size in MB

    Args:
        model: PyTorch model

    Returns:
        Size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# Test the model
if __name__ == "__main__":
    # Create model
    model = ColorClassifierCNN(num_classes=6)

    # Print architecture
    print("="*60)
    print("COLOR CLASSIFIER CNN ARCHITECTURE")
    print("="*60)
    print(model)
    print()

    # Count parameters
    total_params = model.count_parameters()
    model_size = get_model_size_mb(model)

    print("="*60)
    print("MODEL STATISTICS")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print()

    # Test with dummy input
    print("="*60)
    print("FORWARD PASS TEST")
    print("="*60)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, 64)
    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (raw logits):\n{output}")
    print()

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)
    print(f"Probabilities:\n{probabilities}")
    print()

    # Get predictions
    predictions = torch.argmax(probabilities, dim=1)
    print(f"Predicted classes: {predictions}")
    print()

    # Check if model meets requirements
    print("="*60)
    print("REQUIREMENTS CHECK")
    print("="*60)
    size_ok = "PASS" if model_size < 5.0 else "FAIL (needs optimization)"
    params_ok = "PASS" if total_params < 1_000_000 else "WARNING (high param count)"
    shape_ok = "PASS" if output.shape == (batch_size, 6) else "FAIL"

    print(f"Model size < 5MB: {size_ok} ({model_size:.2f} MB)")
    print(f"Parameters < 1M: {params_ok} ({total_params:,})")
    print(f"Output shape correct: {shape_ok}")
    print()

    if model_size >= 5.0:
        print("NOTE: Model is larger than 5MB target.")
        print("      This is OK - will be reduced with quantization/pruning")
        print()

    print("Model ready for training!")
