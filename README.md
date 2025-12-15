# Rubik's Cube Scanner & Solver

A complete end-to-end computer vision and AI system that scans a Rubik's Cube using a camera or image files, identifies the colors of each facelet using a CNN classifier, and solves the cube using the Kociemba two-phase algorithm or IDA* with pattern database heuristics. Features cross-platform support (Windows, macOS, Linux, NVIDIA Jetson), GPU acceleration, and interactive 3D solution animation.

## Features

### Computer Vision Pipeline
- **5 Segmentation Algorithms**: Multiple approaches for extracting facelets from cube images
  - Grid Division: Fast grid-based subdivision for centered cubes
  - Contour-Perspective: Handles tilted/rotated cubes with perspective correction
  - Contour-Neighbor: Robust neighbor validation (default, most reliable)
  - Canny-Square: Edge-based square detection for high-contrast images
  - Brightness-Otsu: Adaptive thresholding for varied lighting conditions

- **20+ Preprocessing Methods**: Comprehensive image enhancement options
  - Filters: bilateral, morphological operations, median, gaussian blur
  - Contrast: CLAHE (LAB/HSV), histogram equalization, contrast stretch, gamma correction
  - Color: saturation boost, white balance
  - Combined pipelines for optimal results

- **CNN-Based Color Classification**: Deep learning model trained on synthetic data to classify 6 Rubik's Cube colors (white, yellow, red, orange, blue, green) with confidence scores

### Solving Algorithms
- **Kociemba Solver**: Fast two-phase algorithm (typically < 1 second)
- **IDA* Solver**: Iterative Deepening A* with pattern database heuristics
  - Corner orientation PDB (2,187 states)
  - Full corner PDB (88,179,840 states)

### Advanced Features
- **Adaptive Evaluation**: Intelligently selects best preprocessing combinations using historical metrics
- **Two-Result Confirmation**: Requires two matching valid results for reliable cube state detection
- **Orientation Correction**: Automatically detects and fixes rotated/flipped faces using edge constraints
- **3D Solution Animation**: Interactive step-by-step visualization with human-readable move instructions
- **GPU Acceleration**: NVIDIA VPI support on Jetson for faster preprocessing

### Platform Support
| Platform | Camera | GPU Acceleration | Display |
|----------|--------|------------------|---------|
| NVIDIA Jetson | IMX219 camera support | VPI acceleration | Native OpenCV |
| Windows | File-based only | CPU | Native OpenCV |
| macOS | File-based only | CPU | Native OpenCV |
| Linux | File-based only | CPU | X11/Wayland |

## Project Structure

```
CS-480-Signature-Assignment/
├── main.py                              # Main menu-driven application
├── RubiksCubeSolver.py                  # High-level solver orchestration
│
├── Segmentation
│   ├── Segmenter.py                     # Segmenter factory/registry
│   ├── FaceletSegmenterGridDivision.py  # V1: Grid-based subdivision
│   ├── FaceletSegmenterContourPerspective.py  # V2: Perspective correction
│   ├── FaceletSegmenterContourNeighbor.py     # V3: Neighbor validation (default)
│   ├── FaceletSegmenterCannySquare.py   # V4: Canny edge detection
│   └── FaceletSegmenterBrightnessOtsu.py      # V5: Otsu thresholding
│
├── Color Classification
│   ├── ColorClassifierCNN.py            # CNN model architecture
│   └── FaceletColorClassifier.py        # Classifier wrapper with batch inference
│
├── Preprocessing
│   ├── ImagePreprocessor.py             # CPU-based preprocessing (20+ methods)
│   └── GPUImagePreprocessor.py          # GPU-accelerated preprocessing (Jetson)
│
├── Solvers
│   ├── IDASolver.py                     # IDA* and Kociemba solvers
│   ├── Cube.py                          # Cube state representation
│   └── Facelet_to_Cube.py               # Facelet-to-cube state conversion
│
├── Evaluation & Metrics
│   ├── cube_evaluation.py               # Cube state validation functions
│   ├── adaptive_evaluator.py            # Smart preprocessing selection
│   ├── PreprocessorMetrics.py           # Historical metrics tracking
│   └── CubeOrientationCorrector.py      # Face orientation correction
│
├── Visualization
│   ├── CubeRenderer.py                  # 3D cube rendering & animation
│   └── DisplayManager.py                # Cross-platform display abstraction
│
├── Camera
│   └── JetsonCamera.py                  # Jetson IMX219 camera interface
│
├── models/                              # Trained model checkpoints
│   ├── best_model.pth                   # PyTorch model
│   └── best_model.onnx                  # ONNX model for faster inference
│
├── datasets/                            # Training and test data
│   ├── training_dataset/                # Synthetic training images
│   └── real_facelets/                   # Real-world facelet samples
│
├── pdb_cache/                           # Pattern database cache files
├── log/                                 # Debug logs
│
└── tools/                               # Training and utility scripts
    ├── train_color_classifier.py        # Train CNN model
    ├── generate_full_dataset.py         # Generate synthetic training data
    ├── preprocessing_harness.py         # Test preprocessing methods
    ├── cube_detection_harness.py        # Test detection algorithms
    ├── analyze_preprocessor_metrics.py  # Analyze historical metrics
    └── ...
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/johnjezl/CS-480-Signature-Assignment.git
cd CS-480-Signature-Assignment
```

### 2. Install Dependencies

**Option A: Using pip (Recommended)**
```bash
pip install -r requirements.txt
```

Then install PyTorch based on your platform:
- **Windows/Linux with CUDA**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- **Windows/Linux CPU-only**: `pip install torch torchvision`
- **macOS**: `pip install torch torchvision`
- **Jetson**: PyTorch is pre-installed with JetPack (do NOT reinstall via pip)

**Option B: Using the installer script**
```bash
python install_dependencies.py
```

### Dependencies
- `numpy>=1.20.0` - Numerical computing
- `opencv-python>=4.5.0` - Computer vision
- `Pillow>=9.0.0` - Image I/O
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Vision models
- `scikit-learn>=1.0.0` - ML utilities
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `tqdm>=4.60.0` - Progress bars

## Usage

### Command Line Options

```bash
python main.py [OPTIONS]
```

**Core Options:**
| Option | Description |
|--------|-------------|
| `--segmenter NAME` | Segmentation algorithm (default: contour-neighbor) |
| `--no-display` | Suppress image display |
| `--rotate` | Rotate camera images 180° (inverted mounting) |
| `--debug` | Enable debug logging to log/debug.log |

**Animation Options:**
| Option | Description |
|--------|-------------|
| `--no-animation` | Disable solution animation |
| `--no-step-by-step` | Continuous animation (no pauses) |

**Preprocessing Options:**
| Option | Description |
|--------|-------------|
| `--segmenter-preprocess METHOD` | Single preprocessing method for segmentation |
| `--cc-preprocess METHOD` | Single preprocessing method for color classification |
| `--all-segmenter-preprocess` | Try all segmentation preprocessing methods |
| `--all-cc-preprocess` | Try all color classification preprocessing methods |
| `--force-centers` | Force center colors to expected values |

**Advanced Options:**
| Option | Description |
|--------|-------------|
| `--nogpu` | Disable GPU acceleration (use CPU only) |
| `--no-adaptive` | Disable adaptive evaluation |

### Available Segmenters

```bash
python main.py --segmenter <NAME>
```

| Name | Algorithm | Best For |
|------|-----------|----------|
| `grid-division` | Grid-based subdivision | Centered, axis-aligned cubes |
| `contour-perspective` | Contour + perspective correction | Tilted or rotated cubes |
| `contour-neighbor` | Contour + neighbor validation | General use (default) |
| `canny-square` | Canny edge + square detection | High-contrast images |
| `brightness-otsu` | Otsu thresholding on brightness | Varied lighting, dark cube plastic |

### Available Preprocessing Methods

```bash
python main.py --segmenter-preprocess <METHOD> --cc-preprocess <METHOD>
```

**Categories:**
- **None**: `none`
- **Bilateral Filters**: `bilateral`, `bilateral-strong`
- **CLAHE**: `clahe-lab`, `clahe-hsv`
- **Edge/Detail**: `unsharp`, `morph-open`, `morph-close`
- **Histogram**: `histeq`, `contrast-stretch`
- **Color Adjustments**: `satboost`, `satboost-mild`, `white-balance`
- **Gamma Correction**: `gamma-bright`, `gamma-dark`
- **Smoothing**: `median`, `gaussian`
- **Combined**: `bilateral-clahe`, `bilateral-sat`, `clahe-sat`, `full-pipeline`

### Operation Modes

**Mode 1: Single Face (File)**
- Load a single cube face image
- Classify the 9 facelet colors
- Display results with confidence scores

**Mode 2: Full Cube (File)**
- Load 6 face images from a directory
- Required files: `up.jpg`, `down.jpg`, `front.jpg`, `back.jpg`, `left.jpg`, `right.jpg`
- Validate cube state (9 of each color)
- Apply orientation correction
- Solve and optionally animate the solution

**Mode 3: Single Face (Camera)** - Jetson Only
- Capture one face via IMX219 camera
- Same processing as Mode 1

**Mode 4: Full Cube (Camera)** - Jetson Only
- Capture all 6 faces with guided prompts
- Same processing as Mode 2

### Examples

**Basic usage with default settings (adaptive mode):**
```bash
python main.py
```

**Use brightness-otsu segmenter with saturation boost:**
```bash
python main.py --segmenter brightness-otsu --cc-preprocess satboost
```

**Try all preprocessing combinations:**
```bash
python main.py --all-segmenter-preprocess --all-cc-preprocess
```

**Disable animation and run in headless mode:**
```bash
python main.py --no-display --no-animation
```

**Jetson with rotated camera:**
```bash
python main.py --rotate --segmenter contour-neighbor
```

## Solution Animation

When solving a cube, the application displays an interactive 3D animation:

- **Step-by-step mode** (default): Loops each move until you press Enter/Space
- **Continuous mode** (`--no-step-by-step`): Plays through all moves automatically

**Controls** (click the animation window first):
- `Enter` or `Space` - Advance to next move (step-by-step) / Pause/Resume (continuous)
- `Q` - Skip animation

Each step shows:
- 3D cube visualization with the current move animating
- Move counter (e.g., "Move 5/20: R'")
- Human-readable instruction (e.g., "Rotate RIGHT face 90° counter-clockwise")

## Training the Color Classifier

### 1. Generate Synthetic Training Data
```bash
python tools/generate_full_dataset.py
```

This creates synthetic facelet images with various color palettes, lighting conditions, and augmentations.

### 2. Train the CNN Model
```bash
python tools/train_color_classifier.py
```

The trained model is saved to `models/best_model.pth`.

### 3. Test on Real Data
```bash
python tools/test_on_real_data.py
```

## CNN Architecture

The `ColorClassifierCNN` is a lightweight convolutional neural network for 64x64 RGB facelet images:

- 3 convolutional blocks with batch normalization and dropout
- Global average pooling
- Fully connected classifier head
- 6-class output (white, yellow, red, orange, blue, green)
- Model size: < 5MB

## Face Orientation Guide

When capturing faces, use this standard orientation:

| Face | Center Color | Top Edge Color |
|------|--------------|----------------|
| Up | Yellow | Green |
| Down | White | Blue |
| Front | Blue | Yellow |
| Back | Green | Yellow |
| Left | Orange | Yellow |
| Right | Red | Yellow |

The orientation corrector will automatically detect and fix most orientation errors using edge constraints.

## Tools Reference

| Tool | Description |
|------|-------------|
| `train_color_classifier.py` | Train the CNN color classifier |
| `generate_full_dataset.py` | Generate synthetic training data |
| `preprocessing_harness.py` | Test preprocessing methods on images |
| `cube_detection_harness.py` | Evaluate cube detection algorithms |
| `preprocessing_detection_harness.py` | Combined preprocessing + detection testing |
| `analyze_preprocessor_metrics.py` | Analyze historical preprocessing metrics |
| `benchmark_preprocessor.py` | Performance profiling of preprocessing |
| `SyntheticFaceletGenerator.py` | Generate synthetic facelet images |
| `analyze_facelet_palette.py` | Extract color palettes from real facelets |
| `test_on_real_data.py` | Evaluate classifier on real images |
| `interactive_facelet_extrator_tool.py` | Manually extract facelets from images |
| `visualize_facelets.py` | Display detected facelets with colors |
| `resize_images.py` | Batch resize images |

## Troubleshooting

### "GPU: Not available" on Jetson
- Ensure VPI is installed with JetPack
- Check that `/opt/nvidia/vpi2/` exists
- Run with `--debug` to see detailed GPU detection logs

### Animation window unresponsive
- Click the animation window to give it keyboard focus
- Press 'Q' to skip if needed

### Segmentation fails
- Try a different segmenter: `--segmenter brightness-otsu`
- Enable preprocessing: `--segmenter-preprocess bilateral`
- Use adaptive mode (default) to automatically find best settings

### Color misclassification
- Try `--cc-preprocess satboost` to enhance color saturation
- Use `--force-centers` to constrain center colors
- Enable `--all-cc-preprocess` to find the best preprocessing


## Authors

- John Jezl
- Greg Nott
- Jon Rocha
- Janice Bargoria
