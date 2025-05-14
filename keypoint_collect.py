import cv2
import mediapipe as mp
import os
import numpy as np
import json
from function import *

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('TRAIN_DATASET_HAND')

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the JSON file path relative to the script's directory
JSON_FILE_PATH = os.path.join(BASE_DIR, 'dataset', 'WLASL100_train.json')

def load_glosses_and_urls(json_file_path):
    """Load glosses and group their corresponding video URLs from the JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    gloss_to_urls = {}
    for entry in data:
        gloss = entry['gloss']
        urls = [instance['url'] for instance in entry['instances']]
        gloss_to_urls[gloss] = urls
    return gloss_to_urls

def setup_video_capture(video_url):
    """Initialize video capture from the given URL."""
    return cv2.VideoCapture(video_url)

def sample_frames(frames, target_len=50):
    """Uniformly sample or pad frames to get exactly target_len frames."""
    current_len = len(frames)
    
    if current_len == 0:
        return []

    if current_len == target_len:
        return frames

    if current_len > target_len:
        # Uniformly sample 50 indices
        indices = np.linspace(0, current_len - 1, target_len).astype(int)
        return [frames[i] for i in indices]

    else:  # current_len < target_len
        # Pad by repeating last frame
        last_frame = frames[-1]
        pad_count = target_len - current_len
        return frames + [last_frame] * pad_count

def extract_video_frames(cap):
    """Read all frames from video and return as list."""
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def process_video(frames, holistic, action, sequence):
    """Process a list of frames (already sampled/padded)."""
    for frame_num, frame in enumerate(frames):
        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks (optional, mainly for visualization)
        draw_styled_landmarks(image, results)
        
        # Export keypoints
        keypoints = extract_keypoints(results)
        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, keypoints)

def collect_data(gloss_to_urls):
    """Main function to collect data for the given glosses and their video URLs."""
    for gloss, urls in gloss_to_urls.items():
        sequence = 0  # Initialize sequence counter for each gloss

        for video_url in urls:
            cap = setup_video_capture(video_url)

            # Set Mediapipe model
            with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5) as holistic:

                # Step 1: Read all frames
                all_frames = extract_video_frames(cap)
                cap.release()

                # Step 2: Sample or pad to 50 frames
                sampled_frames = sample_frames(all_frames, target_len=50)

                # Step 3: Process sampled frames
                process_video(sampled_frames, holistic, gloss, sequence)

                sequence += 1

            cv2.destroyAllWindows()

if __name__ == "__main__":
    gloss_to_urls = load_glosses_and_urls(JSON_FILE_PATH)
    collect_data(gloss_to_urls)
