import cv2
import mediapipe as mp
import os
import numpy as np
from function import *

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect (can be dynamically updated)
actions = np.array(['book'])  # Example gloss (word)

video_url = 'https://aslbricks.org/New/ASL-Videos/book.mp4'


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


def collect_data(actions, video_url):
    """Main function to collect data for the given actions from the video."""
    cap = setup_video_capture(video_url)

    # Set mediapipe model
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            sequence = 0  # Initialize sequence counter

            while True:  # Dynamically process video frames
                frame_num = 0  # Initialize frame counter for each sequence

                while True:  # Process frames for the current sequence
                    # Read feed
                    ret, frame = cap.read()
                    if not ret:  # Break if the video ends
                        break

                    process_frame(frame, holistic, action, sequence, frame_num)
                    frame_num += 1

                    # Break gracefully on 'q' key press
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                # Increment sequence counter
                sequence += 1

                # Break if the video ends
                if not ret:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_data(actions, video_url)