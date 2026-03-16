# Gender and Age Detection (Fixed)

Real-time webcam demo for age/gender prediction using OpenCV + Caffe DNN.

**Fixes applied:**
- Downloaded real models (`gender_net.caffemodel`, `age_net.caffemodel`).
- Fixed input size to 227x227.
- Added **temporal smoothing + face tracking** (5-frame buffer, majority vote) to eliminate flicker.
- Use `--buffer-size N` to adjust stability vs responsiveness.

## 🚀 Quick Start

1. **Deps already installed** (`py -m pip install -r requirements.txt`).

2. Run real-time detection:

```bash
py detect_age_gender.py
```

Press 'q' to quit.

3. **New options:**
```bash
py detect_age_gender.py --buffer-size 10  # More stable
py detect_age_gender.py --verify          # Manual accuracy check
```

## Test

- Static: Add `--image sample.jpg` (future).
- Camera: Stable predictions, no flicker.

Models ready – no dummy mode!
