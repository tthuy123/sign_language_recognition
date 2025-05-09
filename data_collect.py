import cv2
import mediapipe as mp
import os
import numpy as np
import json
from function import *

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

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

# Load actions (glosses) and video URLs
gloss_to_urls = load_glosses_and_urls(JSON_FILE_PATH)

def setup_video_capture(video_url):
    """Initialize video capture from the given URL."""
    return cv2.VideoCapture(video_url)

def process_frame(frame, holistic, action, sequence, frame_num):
    """Process a single frame: detect landmarks, draw them, and save keypoints."""
    # Make detections
    image, results = mediapipe_detection(frame, holistic)

    # Draw landmarks
    draw_styled_landmarks(image, results)
    
    # Export keypoints
    keypoints = extract_keypoints(results)
    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)  # Dynamically create directories
    np.save(npy_path, keypoints)

def collect_data(gloss_to_urls):
    """Main function to collect data for the given glosses and their video URLs."""
    for gloss, urls in gloss_to_urls.items():
        sequence = 0  # Initialize sequence counter for each gloss

        for video_url in urls:
            cap = setup_video_capture(video_url)

            # Set Mediapipe model
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                while True:  # Dynamically process video frames
                    frame_num = 0  # Initialize frame counter for each sequence

                    while True:  # Process frames for the current sequence
                        # Read feed
                        ret, frame = cap.read()
                        if not ret:  # Break if the video ends
                            break

                        process_frame(frame, holistic, gloss, sequence, frame_num)
                        frame_num += 1

                        # Break gracefully on 'q' key press
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()

                    # Increment sequence counter after processing a video
                    sequence += 1

                    # Break if the video ends
                    if not ret:
                        break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    gloss_to_urls = load_glosses_and_urls(JSON_FILE_PATH)
    collect_data(gloss_to_urls)