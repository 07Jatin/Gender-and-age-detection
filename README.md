# Gender and Age Detection

A real-time webcam-based application for detecting gender and estimating age using OpenCV and pre-trained Caffe deep learning models.

## Features

- **Real-time Detection**: Uses webcam feed for live gender and age prediction
- **Stable Predictions**: Implements temporal smoothing and face tracking to eliminate flickering
- **Configurable Buffer**: Adjust buffer size for stability vs responsiveness trade-off
- **Manual Verification**: Option to manually verify predictions for accuracy testing
- **Easy to Use**: Simple command-line interface with optional arguments

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/07Jatin/Gender-and-age-detection.git
   cd Gender-and-age-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the models (if not already present):
   ```bash
   python download_models.py
   ```

## Usage

### Real-time Webcam Detection

Run the main detection script:

```bash
python detect_age_gender.py
```

Press 'q' to quit the application.

### Command-line Options

- `--buffer-size N`: Set the buffer size for temporal smoothing (default: 5). Higher values provide more stable predictions but may be less responsive.
- `--confidence-threshold X`: Minimum softmax confidence for the model to accept a label (default: 0.6). If predictions are below this threshold, the script shows `Unknown` to reduce flicker.
- `--verify`: Enable manual verification mode for testing accuracy.
- `--camera-index N`: Select which webcam to use (useful on systems with multiple cameras).

Examples:

```bash
# More stable predictions
python detect_age_gender.py --buffer-size 10

# Reduce flicker by requiring higher confidence
python detect_age_gender.py --confidence-threshold 0.7

# Manual accuracy check
python detect_age_gender.py --verify
```

## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy
- Pre-trained Caffe models (automatically downloaded)

## Model Details

- **Gender Model**: Binary classification (Male/Female)
- **Age Model**: 8 age ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- Input size: 227x227 pixels
- Framework: Caffe DNN

## How It Works

1. Captures frames from webcam
2. Detects faces using OpenCV's Haar cascades
3. Preprocesses detected faces for model input
4. Runs inference on gender and age models
5. Applies temporal smoothing across frames for stable predictions
6. Displays results with bounding boxes and labels

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is open-source. Please check the license file for details.
