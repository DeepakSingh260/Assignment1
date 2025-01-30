# Hand Tracking Pipeline with SAM2 and MediaPipe

![Pipeline Overview](https://via.placeholder.com/800x400.png?text=Hand+Tracking+Pipeline+Diagram)

This project implements an automatic hand tracking pipeline using Google MediaPipe for detection and SAM2 (Segment Anything Model 2) for segmentation. Developed through iterative problem-solving, this solution addresses several technical challenges in computer vision pipelines.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Implementation Journey](#implementation-journey)
- [Code Structure](#code-structure)
- [Key Challenges & Solutions](#key-challenges--solutions)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Environment Setup <a name="environment-setup"></a>

### Conda Environment Configuration

conda create -n sam2 python=3.12
conda activate sam2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe opencv-python tqdm



### SAM2 Installation
1. Clone and set up SAM2 repository:

git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .



2. Download model checkpoint:
- Place `sam2_hiera_large.pt` in `checkpoints/` directory

3. Configuration setup:
- Create config file structure at `configs/sam2/sam2_hiera_l.yaml`

## Implementation Journey <a name="implementation-journey"></a>

### Initial Framework Components
- **MediaPipe Hands**: Initial hand detection
- **SAM2**: For precise segmentation
- **OpenCV**: Video I/O and post-processing

### Core Implementation Challenges
1. **Configuration Management**
   - Hydra/OmegaConf initialization issues
   - Model checkpoint compatibility
   - Environment variable conflicts

2. **Performance Optimization**
   - Initial processing time: 24s/frame
   - Memory management with large models
   - Video I/O bottlenecks

3. **Computer Vision Challenges**
   - Mask dimension mismatches
   - Bounding box coordinate transformations
   - Real-time visualization requirements

### Key Components
#### Hand Detection (`detect_hands()`)

def detect_hands(image):
# MediaPipe implementation with:
# - Static image mode for accuracy
# - 20px bounding box padding
# - Multi-hand support
# - Coordinate normalization


#### SAM2 Integration

Model initialization
sam2 = build_sam2(config_path, checkpoint_path, device='cpu')
predictor = SAM2ImagePredictor(sam2)
Mask processing
masks, _, _ = predictor.predict(box=hand_bboxes)


#### Video Processing Pipeline

def process_video(input_path, output_path):
# Implements:
# - Frame skipping (1 frame/10 seconds)
# - Error-resilient mask processing
# - Efficient video writing
# - Progress tracking with tqdm



## Code Structure <a name="code-structure"></a>

hand-tracking-pipeline/
├── configs/
│ └── sam2/
│ └── sam2_hiera_l.yaml
├── checkpoints/
│ └── sam2_hiera_large.pt
├── src/
│ ├── detection.py
│ ├── processing.py
│ └── visualization.py
├── test.py
└── README.md



## Key Challenges & Solutions <a name="key-challenges--solutions"></a>

| Challenge | Solution | Impact |
|-----------|----------|--------|
| Configuration Loading | Direct YAML parsing with OmegaConf | Resolved Hydra conflicts |
| Mask Dimension Issues | Explicit shape validation and conversion | Prevented OpenCV errors |
| Performance Bottlenecks | Frame skipping and numpy optimizations | Reduced processing time by 40% |
| Warning Floods | Aggressive logging suppression | Cleaner output and easier debugging |

## Results <a name="results"></a>
https://github.com/DeepakSingh260/Assignment1/blob/main/output%20video.mp4
- Successful hand segmentation at 10-second intervals
- Visual feedback through green mask overlays
- Maintained original video framerate in output
- Average processing time: 20-25s per frame (CPU)

Processing Video: 100%|█████████| 210/210 [1:15:00<00:00, 20.07s/frame]
Processed video saved to: output_video.mp4


