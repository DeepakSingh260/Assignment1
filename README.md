# Hand Tracking Pipeline with SAM2 and MediaPipe

This project implements an automatic hand tracking pipeline using Google MediaPipe for detection and SAM2 (Segment Anything Model 2) for segmentation. Below is the detailed thought process and implementation journey.

## Environment Setup

### Conda Environment

conda create -n sam2 python=3.12
conda activate sam2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe opencv-python tqdm


### SAM2 Installation

1. Cloned SAM2 repository:
   
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .


2. Downloaded `sam2_hiera_large.pt` checkpoint
3. Created config file structure at `configs/sam2/sam2_hiera_l.yaml`

## Implementation Journey

### 1. Initial Framework
- MediaPipe Hands for initial hand detection
- SAM2 for segmentation
- OpenCV for video I/O

Key challenges faced:
- SAM2 configuration issues with Hydra/OmeagConf
- Mask dimension mismatches
- TensorFlow Lite warning floods

### 2. Core Components

#### Hand Detection (`detect_hands()`)
- Used MediaPipe's Hands model with:
  - `static_image_mode=True` for better single frame accuracy
  - 20px padding around detected landmarks
  - Multi-hand support with bounding box aggregation

#### SAM2 Integration
sam2 = build_sam2(config_path, checkpoint_path, device='cpu')
predictor = SAM2ImagePredictor(sam2)

- CPU-only implementation due to CUDA limitations
- Batch processing disabled for memory efficiency

### 3. Video Processing Pipeline
def process_video(input_path, output_path):
  # Frame skipping logic (1 frame/10 seconds)
  # Mask processing with error handling
  # OpenCV video writer with MP4V codec


Key optimizations:
- Frame skipping for practical processing times
- Numpy-based mask blending instead of OpenCV operations
- Aggressive warning suppression for cleaner output

## Code Structure

Key optimizations:
- Frame skipping for practical processing times
- Numpy-based mask blending instead of OpenCV operations
- Aggressive warning suppression for cleaner output

## Code Structure

├── configs/
│ └── sam2/
│ └── sam2_hiera_l.yaml
├── checkpoints/
│ └── sam2_hiera_large.pt
├── test.py
└── README.md



## Results

- Successful hand segmentation at 10-second intervals
- Green overlay visualization on detected hands
- Output video maintains original framerate



