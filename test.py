import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')  

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

def detect_hands(image):
    """Detects hands in the image and returns bounding boxes."""
    h, w, _ = image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_bboxes = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            # Add padding to the bounding box
            padding = 20
            hand_bboxes.append([
                max(0, x_min - padding),
                max(0, y_min - padding),
                min(w, x_max + padding),
                min(h, y_max + padding)
            ])
        return np.array(hand_bboxes, dtype=np.int32)
    return None

def process_video(input_path, output_path):
    """Processes the input video with one frame every 10 seconds."""
    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "configs/sam2/sam2_hiera_l.yaml"
    print(f"Loading model config from: {model_cfg}")

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cpu', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2)

    # Open video
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = int(fps * 10)
    if frame_interval == 0:
        frame_interval = 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % 10 != 0:
                frame_count += 1
                pbar.update(1)
                continue

            hand_bboxes = detect_hands(frame)
            
            if hand_bboxes is not None:
                try:
                    predictor.set_image(frame)
                    masks, _, _ = predictor.predict(box=hand_bboxes)
                    
                    for mask in masks:
                        if mask is not None and mask.any():
                            binary_mask = (mask * 255).astype(np.uint8)
                            if len(binary_mask.shape) == 3:
                                binary_mask = binary_mask[:, :, 0]  # Take first channel
                            
                            binary_mask = cv2.resize(binary_mask, (width, height))
                            print("masking")
                            frame = np.where(
                                binary_mask[..., None] > 128,
                                frame * 0.7 + np.array([0, 255, 0]) * 0.3,
                                frame
                            ).astype(np.uint8)

                except Exception as e:
                    print(f"\nError processing frame: {str(e)}")
                    continue

            out.write(frame)
            frame_count += 1
            pbar.update(1)

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()
    sys.stderr = sys.__stderr__  # Restore stderr
    print(f"\nProcessed video saved to: {output_path}")

# Usage
input_video = "test.mp4"
output_video = "output_video.mp4"
process_video(input_video, output_video)
