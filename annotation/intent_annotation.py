"""
Person Intent Tracker

This script processes videos to annotate object intents in frames.
Key features:
- Handles videos from moving vehicle perspective
- Scales bounding boxes (video frames are half size of reference)
- Annotates vertical intent (along road) and lateral intent (across frame)
- Considers motion relative to road, not the moving vehicle
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm
import json
import requests
import tempfile
import random
import os
from collections import deque
from itertools import islice
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize models
yolo_model = YOLO('yolov8x.pt')
cycle_model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
cycle_model = cycle_model.to(device)
cycle_model.eval()

def detect_cycles(frame: np.ndarray, confidence_threshold: float = 0.5) -> List[List[float]]:
    """
    Detect cycles using Faster R-CNN
    Returns list of [x1, y1, x2, y2, confidence] for each detection
    """
    # Convert frame to tensor and move to GPU if available
    image = F.to_tensor(frame).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        prediction = cycle_model(image)

    # Extract bicycle detections (class 2 in COCO)
    boxes = []
    for box, label, score in zip(prediction[0]['boxes'],
                                prediction[0]['labels'],
                                prediction[0]['scores']):
        if label == 2 and score > confidence_threshold:  # bicycle class
            x1, y1, x2, y2 = box.cpu().numpy()
            boxes.append([x1, y1, x2, y2, float(score)])

    return boxes

def check_cyclist_association(person_box: List[float],
                            cycle_box: List[float],
                            max_vertical_offset: float = 160.0,
                            min_vertical_offset: float = 0.0,
                            min_overlap_ratio: float = 0.3) -> bool:
    """
    Check if person and cycle detections likely form a cyclist using centroids
    """
    # Unpack boxes
    px1, py1, px2, py2 = person_box[:4]
    cx1, cy1, cx2, cy2 = cycle_box[:4]

    # Calculate centroids and box heights
    person_centroid_y = (py1 + py2) / 2
    cycle_centroid_y = (cy1 + cy2) / 2
    person_height = py2 - py1
    cycle_height = cy2 - cy1

    # Check vertical alignment (person above cycle)
    vertical_offset = person_centroid_y - cycle_centroid_y

    # Adjust max offset based on object sizes
    adjusted_max_offset = max(max_vertical_offset, (person_height + cycle_height) / 2)

    # Person should be above cycle (negative offset) but not too far
    # Also ensure minimum separation to avoid false positives
    if not (-adjusted_max_offset <= vertical_offset):
        return False

    # Check horizontal overlap
    overlap_x1 = max(px1, cx1)
    overlap_x2 = min(px2, cx2)

    if overlap_x2 <= overlap_x1:
        return False

    # Calculate overlap ratio
    overlap_width = overlap_x2 - overlap_x1
    person_width = px2 - px1
    cycle_width = cx2 - cx1
    overlap_ratio = overlap_width / min(person_width, cycle_width)
    return overlap_ratio >= min_overlap_ratio

def update_cyclist_tracks(track_histories: Dict[int, Dict],
                         current_frame: int,
                         person_detections: List[Tuple[List[float], int]],
                         cycle_detections: List[List[float]],
                         cyclist_memory: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Update tracking with cyclist combinations
    """
    # Update cyclist memory with new associations
    for person_box, person_id in person_detections:
        for cycle_box in cycle_detections:
            if check_cyclist_association(person_box, cycle_box):
                if person_id not in cyclist_memory:
                    cyclist_memory[person_id] = {
                        'frames_as_cyclist': 0,
                        'last_cycle_box': cycle_box,
                        'confirmed': False
                    }

                cyclist_memory[person_id]['frames_as_cyclist'] += 1
                cyclist_memory[person_id]['last_cycle_box'] = cycle_box

                # Confirm as cyclist after consistent detection
                if cyclist_memory[person_id]['frames_as_cyclist'] >= 5:
                    cyclist_memory[person_id]['confirmed'] = True
                    if person_id in track_histories:
                        track_histories[person_id]['class'] = 'Cyclists'
                break

    # Remove old entries and unconfirm if no recent associations
    to_remove = []
    for person_id in cyclist_memory:
        if person_id not in [pid for _, pid in person_detections]:
            cyclist_memory[person_id]['frames_as_cyclist'] -= 1
            if cyclist_memory[person_id]['frames_as_cyclist'] <= 0:
                to_remove.append(person_id)
                if person_id in track_histories and track_histories[person_id]['class'] == 'Cyclists':
                    track_histories[person_id]['class'] = 'Pedestrians'

    for person_id in to_remove:
        del cyclist_memory[person_id]

    return track_histories

class IntentAnalyzer:
    def __init__(self, frame_size: Tuple[int, int], motion_threshold: float = 0.2):
        """
        Initialize intent analyzer

        Args:
            frame_size: Size of the video frames (width, height)
            motion_threshold: Minimum pixel movement to consider as motion
        """
        self.frame_size = frame_size
        self.position_threshold = frame_size[0] / 3  # Center line for left/right position
        self.motion_threshold = motion_threshold
        self.track_histories = {}  # Store centroid history for each track
        self.scale_factor = 0.5  # Video frames are half size of reference
        self.camera_motion = []   # Store camera centroids for each frame

    def set_camera_motion(self, camera_centroids: list):
        """
        Set the camera centroids for the video (should be called before intent analysis)
        """
        self.camera_motion = camera_centroids

    def scale_bbox(self, bbox: List[int]) -> List[int]:
        """Scale bounding box coordinates to match video frame size"""
        return [int(coord * self.scale_factor) for coord in bbox]

    def update_track_history(self, track_id: int, centroid: Tuple[int, int]):
        """Update tracking history for a specific track ID"""
        if track_id not in self.track_histories:
            self.track_histories[track_id] = []
        self.track_histories[track_id].append(centroid)

    def determine_position(self, centroid: Tuple[int, int]) -> str:
        """Determine object position relative to ego vehicle"""
        if centroid[0] < self.position_threshold:
            return "Left of ego vehicle"
        if centroid[0] > self.frame_size[0] - self.position_threshold:
            return "Right of ego vehicle"
        return "Front of ego vehicle"

    def determine_intent(self, track_id: int, cam_dx, cam_dy) -> List[str]:
        """
        Determine vertical and lateral intent based on track history and camera motion.
        Returns [lateral_intent, vertical_intent]
        """
        history = list(self.track_histories.get(track_id, []))
        if len(history) < 3:
            return ["stationary", "stationary"]

        # If camera motion is available, subtract it from object centroids
        if self.camera_motion and len(self.camera_motion) >= len(history) and False:
            rel_history = [
                (obj[0] - cam[0], obj[1] - cam[1])
                for obj, cam in zip(history, self.camera_motion[-len(history):])
            ]
        else:
            rel_history = history

        # Net movement (first to last)
        net_dx = rel_history[-1][0] - rel_history[0][0]
        net_dy = rel_history[-1][1] - rel_history[0][1]


        # Calculate motion over multiple windows to catch subtle movements
        windows = [(0, len(rel_history)//2), (len(rel_history)//2, len(rel_history))]
        dx_values = []
        dy_values = []
        for start, end in windows:
            if end - start < 2:
                continue
            window = rel_history[start:end]
            dx = [window[i+1][0] - window[i][0] for i in range(len(window)-1)]
            dy = [window[i+1][1] - window[i][1] for i in range(len(window)-1)]
            dx_values.extend(dx)
            dy_values.extend(dy)
        if not dx_values or not dy_values:
            return ["stationary", "stationary"]

        # Calculate average and consistency of motion
        avg_dx = np.mean(dx_values)
        avg_dy = np.mean(dy_values)
        std_dx = np.std(dx_values)
        std_dy = np.std(dy_values)

        # Weighted combination: net movement gets higher weight
        final_dx = 1 * net_dx + 0 * avg_dx - cam_dx
        final_dy = 1 * net_dy + 0 * avg_dy - cam_dy

        # Determine lateral intent (horizontal motion)
        lateral_intent = "stationary"
        # if abs(final_dx) > self.motion_threshold and std_dx < abs(final_dx) * 2:
        if abs(final_dx) > self.motion_threshold:
            lateral_intent = "goes to the right" if final_dx > 0 else "goes to the left"

        # Determine vertical intent (along road)
        # Moving up in the frame (negative dy) means moving away from ego vehicle
        vertical_intent = "stationary"
        if abs(final_dy) > self.motion_threshold:
            vertical_intent = "moves away from ego vehicle" if final_dy < 0 else "moves towards ego vehicle"
        return [lateral_intent, vertical_intent]

    def generate_description(self, intent: List[str]) -> str:
        """Generate human-readable description from intent"""
        lateral, vertical = intent
        return f"Lateral: {lateral}, Vertical: {vertical}"

def download_video(url: str) -> str:
    """Downloads video content from URL and saves to a temporary file"""
    # print(f"Downloading video from {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download video: {response.status_code}")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    for chunk in response.iter_content(chunk_size=1024*1024):
        if chunk:
            temp_file.write(chunk)
    temp_file.close()
    return temp_file.name

def load_json_data(file_path: str) -> dict:
    """Load JSON data from file"""
    print(f"Loading JSON data from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_data(data: dict, file_path: str):
    """Save JSON data to file"""
    print(f"Saving JSON data to {file_path}")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_centroid(box: List[int]) -> Tuple[int, int]:
    """Calculate centroid from bounding box coordinates"""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def estimate_camera_motion(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, float]:
    """
    Estimate camera motion between two frames using optical flow.
    Returns the average motion vector (dx, dy).
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    feature_params = dict(maxCorners=100,
                         qualityLevel=0.3,
                         minDistance=7,
                         blockSize=7)

    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if p0 is None:
        return 0, 0

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Calculate motion vectors
        motion_vectors = good_new - good_old
        # Get median motion to remove outliers
        if len(motion_vectors) > 0:
            median_motion = np.median(motion_vectors, axis=0)
            return median_motion[0], median_motion[1]

    return 0, 0

def calculate_displacement(points: List[Tuple[float, float]]) -> Tuple[float, Tuple[float, float], Tuple[float, float], float, float]:
    """Calculate net displacement between first and last point"""
    if len(points) < 2:
        return 0.0, (0.0, 0.0), (0.0, 0.0), 0.0, 0.0

    start = np.array(points[0], dtype=np.float32)
    end = np.array(points[-1], dtype=np.float32)

    displacement = float(np.linalg.norm(end - start))
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])

    return displacement, tuple(start), tuple(end), dx, dy

def predict_next_position(track: Dict, num_frames: int = 1) -> Tuple[np.ndarray, float]:
    """
    Predict future position based on recent motion with confidence score
    """
    if len(track['centroids']) < 2:
        return np.array(track['centroids'][-1]), 0.5

    # Get recent positions
    recent = np.array(track['centroids'][-5:])
    if len(recent) < 2:
        return recent[-1], 0.5

    # Calculate velocities for each consecutive pair
    velocities = recent[1:] - recent[:-1]

    # Calculate acceleration (change in velocity)
    if len(velocities) > 1:
        accelerations = velocities[1:] - velocities[:-1]
        avg_acceleration = np.mean(accelerations, axis=0)
    else:
        avg_acceleration = np.zeros(2)

    # Get last velocity
    last_velocity = velocities[-1]

    # Predict velocity considering acceleration
    predicted_velocity = last_velocity + (avg_acceleration * num_frames)

    # Predict position using physics equations
    last_pos = recent[-1]
    predicted_pos = last_pos + (predicted_velocity * num_frames) + (0.5 * avg_acceleration * num_frames * num_frames)

    # Calculate prediction confidence based on motion consistency
    velocity_consistency = 1.0 - min(1.0, np.std(velocities) / (np.linalg.norm(last_velocity) + 1e-6))
    temporal_factor = np.exp(-0.1 * num_frames)  # Confidence decreases with prediction distance
    confidence = velocity_consistency * temporal_factor

    return predicted_pos, confidence

def calculate_appearance_similarity(frame1: np.ndarray, frame2: np.ndarray,
                                 box1: List[float], box2: List[float]) -> float:
    """
    Calculate appearance similarity between two object patches
    """
    try:
        # Extract patches
        x1, y1 = int(box1[0]), int(box1[1])
        x2, y2 = int(box1[2]), int(box1[3])
        patch1 = frame1[y1:y2, x1:x2]

        x1, y1 = int(box2[0]), int(box2[1])
        x2, y2 = int(box2[2]), int(box2[3])
        patch2 = frame2[y1:y2, x1:x2]

        # Resize patches to same size
        size = (64, 64)
        patch1 = cv2.resize(patch1, size)
        patch2 = cv2.resize(patch2, size)

        # Convert to grayscale
        gray1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

        # Calculate histogram similarity
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, similarity)  # Ensure non-negative
    except:
        return 0.0

def should_merge_tracks(track1: Dict, track2: Dict,
                       max_frame_gap: int = 15,
                       max_spatial_dist: float = 100.0) -> bool:
    """
    Determine if two tracks should be merged based on multiple metrics
    """
    # Check if same class
    if track1['class'] != track2['class']:
        return False

    # Get end frame of track1 and start frame of track2
    end_frame = track1['frame_nums'][-1]
    start_frame = track2['frame_nums'][0]

    # Check temporal proximity
    frame_gap = start_frame - end_frame
    if frame_gap <= 0 or frame_gap > max_frame_gap:
        return False

    # 1. Check spatial proximity with prediction
    predicted_pos, confidence = predict_next_position(track1, frame_gap)
    start_point = np.array(track2['centroids'][0])
    spatial_dist = np.linalg.norm(predicted_pos - start_point)

    # Adjust max_spatial_dist based on prediction confidence and frame gap
    adjusted_max_dist = max_spatial_dist * (1 + (1 - confidence) * 0.5) * (1 + frame_gap / max_frame_gap)

    if spatial_dist > adjusted_max_dist:
        return False

    # 2. Check velocity consistency with more tolerance for short gaps
    if len(track1['centroids']) >= 2 and len(track2['centroids']) >= 2:
        vel1 = np.array(track1['centroids'][-1]) - np.array(track1['centroids'][-2])
        vel2 = np.array(track2['centroids'][1]) - np.array(track2['centroids'][0])
        vel_diff = np.linalg.norm(vel1 - vel2)
        vel_threshold = adjusted_max_dist * 0.75  # More lenient velocity threshold
        if vel_diff > vel_threshold:
            return False

    # 3. Calculate track direction similarity with tolerance
    if len(track1['centroids']) >= 2 and len(track2['centroids']) >= 2:
        dir1 = np.array(track1['centroids'][-1]) - np.array(track1['centroids'][0])
        dir2 = np.array(track2['centroids'][-1]) - np.array(track2['centroids'][0])
        dir_similarity = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-6)
        if dir_similarity < -0.5:  # More lenient direction threshold
            return False

    # Combine all metrics into a score with confidence weighting
    spatial_score = 1.0 - (spatial_dist / adjusted_max_dist)
    temporal_score = 1.0 - (frame_gap / max_frame_gap)

    # Weight the scores based on prediction confidence
    final_score = (0.6 * spatial_score + 0.4 * temporal_score) * (0.5 + 0.5 * confidence)

    # More lenient threshold for short gaps
    threshold = 0.2 if frame_gap <= 3 else 0.3

    return final_score > threshold

def merge_tracks(track1: Dict, track2: Dict) -> Dict:
    """
    Merge two track histories into one.
    """
    # print(f"Merging tracks {track1['centroids']} \n and {track2['centroids']} to get {track1['centroids'] + track2['centroids']}\n")
    return {
        'boxes': track1['boxes'] + track2['boxes'],
        'centroids': track1['centroids'] + track2['centroids'],
        'frame_nums': track1['frame_nums'] + track2['frame_nums'],
        'class': track1['class']  # They should be the same class
    }

def link_broken_tracks(track_histories: Dict[int, Dict],
                      max_frame_gap: int = 15,
                      max_spatial_dist: float = 100.0) -> Dict[int, Dict]:
    """
    Link tracks that likely belong to the same object.

    Args:
        track_histories: Dictionary of track histories
        max_frame_gap: Maximum frame gap to consider for linking
        max_spatial_dist: Maximum spatial distance to consider for linking

    Returns:
        Dictionary of merged track histories
    """
    # Remove camera from consideration
    if len(track_histories.keys()) == 1:
        return track_histories
    camera_track = track_histories.pop('camera', None)

    # Sort tracks by start frame
    sorted_tracks = sorted(track_histories.items(),
                         key=lambda x: x[1]['frame_nums'][0])


    merged_tracks = {}
    used_tracks = set()
    next_track_id = max(track_histories.keys()) + 1

    # Try to link tracks
    for i, (track_id1, track1) in enumerate(sorted_tracks):
        if track_id1 in used_tracks:
            continue

        current_track = track1
        used_tracks.add(track_id1)

        # Look for tracks to merge with current_track
        for track_id2, track2 in sorted_tracks[i+1:]:
            if track_id2 in used_tracks:
                continue

            if should_merge_tracks(current_track, track2, max_frame_gap, max_spatial_dist):
                current_track = merge_tracks(current_track, track2)
                used_tracks.add(track_id2)

        merged_tracks[next_track_id] = current_track
        next_track_id += 1

    # Add back camera track if it existed
    if camera_track:
        merged_tracks['camera'] = camera_track

    return merged_tracks

def process_video(video_url: str, intent_analyzer: IntentAnalyzer,
                 conf_threshold: float = 0.25) -> Dict[int, Dict]:
    """
    Process video to track objects and analyze their intent
    Returns dictionary mapping object IDs to their tracking histories
    """
    # Download and open video
    video_path = download_video(video_url)
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height

    track_histories = {}  # Store tracking histories
    camera_motion = []   # Store camera motion vectors
    prev_frame = None
    frame_count = 0
    cyclist_memory = {}  # Store cyclist detection history

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Estimate camera motion if we have a previous frame
        if prev_frame is not None:
            dx, dy = estimate_camera_motion(prev_frame, frame)
            camera_motion.append((dx, dy))
        prev_frame = frame.copy()

        # Run YOLOv8 tracking for persons
        results = yolo_model.track(frame, persist=True, conf=conf_threshold, classes=[0], verbose=False, tracker="botsort.yaml")  # person class only

        # Get cycle detections
        cycle_boxes = detect_cycles(frame)

        person_detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Convert boxes to xyxy format for person detections
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                xyxy = [x-w/2, y-h/2, x+w/2, y+h/2]
                person_detections.append((xyxy, track_id))

                # Filter out unrealistic detections
                box_area = w * h
                if box_area < 0.001 * frame_area or box_area > 0.9 * frame_area:
                    continue

                if track_id not in track_histories:
                    track_histories[track_id] = {
                        'boxes': [],
                        'centroids': [],
                        'frame_nums': [],
                        'class': 'Pedestrians'
                    }

                track_histories[track_id]['boxes'].append(xyxy)
                track_histories[track_id]['centroids'].append((float(x), float(y)))
                track_histories[track_id]['frame_nums'].append(frame_count)

                # Draw centroid on frame
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(frame, f'ID: {track_id}', (int(x), int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update cyclist tracks
        track_histories = update_cyclist_tracks(track_histories, frame_count,
                                              person_detections, cycle_boxes,
                                              cyclist_memory)

        # Draw cycle detections
        for box in cycle_boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Display frame
        # display_frame_with_grid(frame)
        frame_count += 1

    cap.release()
    os.remove(video_path)

    # Add camera motion to the return dictionary
    track_histories['camera'] = {
        'centroids': [(0, 0)],
        'frame_nums': [0],
        'class': 'Camera'
    }

    # Accumulate camera motion
    cam_x, cam_y = 0, 0
    for i, (dx, dy) in enumerate(camera_motion, 1):
        cam_x += dx
        cam_y += dy
        track_histories['camera']['centroids'].append((cam_x, cam_y))
        track_histories['camera']['frame_nums'].append(i)

    # Link broken tracks
    track_histories = link_broken_tracks(track_histories,
                                       max_frame_gap=15,
                                       max_spatial_dist=100.0)

    # Calculate and print displacements
    displacements = {}
    for track_id, track_data in track_histories.items():
        # print(f"Trackid: {track_id}Centroids: {track_data['centroids']}")
        displacement, start, end, dx, dy = calculate_displacement(track_data['centroids'])
        displacements[track_id] = {
            'class': track_data['class'],
            'displacement': displacement,
            'start_point': track_data['centroids'][0],
            'end_point': track_data['centroids'][-1],
            'start_frame': track_data['frame_nums'][0],
            'end_frame': track_data['frame_nums'][-1],
            'dx': dx,
            'dy': dy,
            'start': start,
            'end': end
        }

    

    yolo_model.predictor.trackers[0].reset()
    return track_histories

def display_frame_with_grid(frame: np.ndarray):
    """
    Display the frame with a grid and x, y coordinates using cv2_imshow.
    """
    height, width, _ = frame.shape

    # Define grid step
    grid_step_x = width // 10  # 10 intervals on x axis
    grid_step_y = height // 10  # 10 intervals on y axis

    # Draw grid lines
    for x in range(0, width, grid_step_x):
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)  # Vertical lines
    for y in range(0, height, grid_step_y):
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)  # Horizontal lines

    # Add x and y axis labels (simple numbering)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, width, grid_step_x):
        cv2.putText(frame, str(x), (x + 2, height - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, height, grid_step_y):
        cv2.putText(frame, str(y), (5, y + 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame with the grid
    # cv2_imshow(frame)

def greedy_match(cost_matrix, track_ids, input_ids):
    # Flatten and sort (i,j) pairs by cost
    flat = [(i,j,c) for i,row in enumerate(cost_matrix)
                 for j,c in enumerate(row)]
    flat.sort(key=lambda x: x[2])
    matches, used_t, used_i = {}, set(), set()
    for i,j,c in flat:
        if i not in used_t and j not in used_i:
            matches[track_ids[i]] = input_ids[j]
            used_t.add(i); used_i.add(j)
    return matches


def best_iou_batch(track_boxes: np.ndarray, input_box: np.ndarray) -> float:
    # track_boxes: (M,4), input_box: (4,)
    x1 = np.maximum(track_boxes[:,0], input_box[0])
    y1 = np.maximum(track_boxes[:,1], input_box[1])
    x2 = np.minimum(track_boxes[:,2], input_box[2])
    y2 = np.minimum(track_boxes[:,3], input_box[3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    track_areas = (track_boxes[:,2] - track_boxes[:,0]) * (track_boxes[:,3] - track_boxes[:,1])
    input_area = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])

    ious = inter / (track_areas + input_area - inter + 1e-8)
    return float(np.max(ious))

def match_objects(type_tracks, input_boxes, iou_thresh=0.3):
    if not type_tracks or not input_boxes:
        return {}
    # prepare data
    input_list, input_ids = [], []
    for bid, bdata in input_boxes.items():
        input_list.append(np.array(bdata['Box'])/2)
        input_ids.append(bid)
    track_ids = list(type_tracks)
    cost = np.zeros((len(track_ids), len(input_list)), dtype=float)

    # build cost matrix
    for i, tid in enumerate(track_ids):
        tb = np.array(type_tracks[tid]['boxes'])  # (M,4)
        for j, ib in enumerate(input_list):
            best = best_iou_batch(tb, ib)
            cost[i,j] = 1 - best

    # if only one candidate, shortcut
    if cost.size == 1:
        return { track_ids[0]: input_ids[0] }

    # Hungarian with fallback
    try:
        r,c = linear_sum_assignment(cost)
        pairs = list(zip(r,c))
    except ValueError:
        pairs = greedy_match(cost, track_ids, input_ids).items()

    # filter by IoU threshold
    matches = {}
    for i,j in pairs:
        if cost[i,j] < 1 - iou_thresh:
            matches[track_ids[i]] = input_ids[j]

    return matches


def find_best_match(tracked_box: List[int], input_boxes: Dict[str, Dict]) -> Optional[str]:
    """Find the best matching input box for a tracked box"""
    best_iou = 0.3  # Lower threshold for matching
    best_id = None

    for obj_id, obj_data in input_boxes.items():
        iou = box_iou(tracked_box, obj_data['Box'])
        if iou > best_iou:
            best_iou = iou
            best_id = obj_id

    return best_id

def convert_bbox_format(bbox):
    """Convert bounding box format if needed"""
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    return [x1, y1, x2, y2]

def remove_duplicate_boxes(input_boxes: Dict[str, Dict], iou_threshold: float = 0.5) -> Dict[str, Dict]:
    """
    Remove duplicate boxes from input data based on IoU threshold
    Keeps the box with intent information if available
    """
    filtered_boxes = {}
    box_ids = list(input_boxes.keys())

    for i, box_id in enumerate(box_ids):
        box1 = input_boxes[box_id]['Box']
        is_duplicate = False

        # Check if this box is a duplicate of any previously kept box
        for kept_id in filtered_boxes:
            box2 = filtered_boxes[kept_id]['Box']

            # Calculate IoU
            intersection_x1 = max(box1[0], box2[0])
            intersection_y1 = max(box1[1], box2[1])
            intersection_x2 = min(box1[2], box2[2])
            intersection_y2 = min(box1[3], box2[3])

            if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                iou = intersection_area / (box1_area + box2_area - intersection_area)
                # print(f"IoU: {iou}")
                if iou > iou_threshold:
                    is_duplicate = True
                    # If current box has intent and kept box doesn't, replace kept box
                    if (input_boxes[box_id]['Intent'] and
                        not filtered_boxes[kept_id]['Intent']):
                        filtered_boxes[kept_id] = input_boxes[box_id]
                    break

        if not is_duplicate:
            filtered_boxes[box_id] = input_boxes[box_id]

    return filtered_boxes

from matplotlib import animation


def animate_optical_flow(video_path, max_frames=100, step=1):
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frames = []

    for _ in range(max_frames):
        for _ in range(step):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        amplification = 5.0
        flow *= amplification
        # Calculate magnitude and angle of optical flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        frames.append(rgb_flow)
        prev_gray = gray

    cap.release()

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    # display(HTML(ani.to_jshtml()))


def get_median_optical_flow_multiple_margins(video_path, point, box_size=(50, 150), margins=[50, 70,  100, 120, 150, 170, 200, 50], max_frames=100, y_shift = False, margin_add = 0):
    """
    For each margin, sums per-pixel flow across frames in adjacent box, then computes the median dx, dy.

    Args:
        video_path (str): Path to video file.
        point (tuple): (x, y) reference point.
        box_size (tuple): Width and height of each box.
        margins (list): List of margin values to test.
        max_frames (int): Max number of frames to process.

    Returns:
        overall_median_dx, overall_median_dy: Median of summed per-pixel motion across all margins.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    x, y = point
    bw, bh = box_size
    bw = w
    margin_medians = []

    margins = [x + margin_add for x in margins]
    for i, margin in enumerate(margins):
        if i == len(margins) - 1:
            y_shift = True
        # Decide box direction
        if x < w / 3:
            box_x = x + margin
        elif x > 2 * w / 3:
            box_x = x - margin - bw
        else:
            box_x = x + margin if random.random() > 0.5 else x - margin - bw

        if y_shift:
            box_y = y + margin - bh

        # Clamp to image bounds
        box_x = int(max(0, min(w - bw, box_x)))
        box_y = int(max(0, min(h - bh, y - bh // 2)))

        # Prepare flow accumulator for the box region
        flow_sum = np.zeros((bh, bw, 2), dtype=np.float32)

        # Rewind and process
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            # Extract and accumulate flow in the box
            flow_box = flow[box_y:box_y+bh, box_x:box_x+bw]
            flow_sum += flow_box

            prev_gray = gray
            frame_count += 1

        # Now compute medians from summed flow
        total_dx = flow_sum[..., 0].flatten()
        total_dy = flow_sum[..., 1].flatten()
        median_dx = float(np.median(total_dx))
        median_dy = float(np.median(total_dy))
        print(f"Margin {margin}: Median Total dx = {median_dx:.2f}, dy = {median_dy:.2f}")
        margin_medians.append((median_dx, median_dy))

    cap.release()

    # Final median over margins
    all_dx = [dx for dx, _ in margin_medians]
    all_dy = [dy for _, dy in margin_medians]
    all_dx.sort()
    all_dy.sort()
    overall_median_dx = float(np.mean(all_dx))
    overall_median_dy = float(np.mean(all_dy))

    print(f"\n→ Overall Median dx = {overall_median_dx:.2f}, dy = {overall_median_dy:.2f}")
    return overall_median_dx, overall_median_dy

def get_median_optical_flow(video_path, point, box_h = 150,  max_frames=100, y_shift = False):
    """
    For each margin, sums per-pixel flow across frames in adjacent box, then computes the median dx, dy.

    Args:
        video_path (str): Path to video file.
        point (tuple): (x, y) reference point.
        box_size (tuple): Width and height of each box.
        margins (list): List of margin values to test.
        max_frames (int): Max number of frames to process.

    Returns:
        overall_median_dx, overall_median_dy: Median of summed per-pixel motion across all margins.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    x, y = point
    bh = box_h
    bw = w
    box_x = 0

    if y_shift:
        box_y = y + 50 - bh

    # Clamp to image bounds
    box_x = int(max(0, min(w - bw, box_x)))
    box_y = int(max(0, min(h - bh, y - bh // 2)))

    # Prepare flow accumulator for the box region
    flow_sum = np.zeros((bh, bw, 2), dtype=np.float32)

    # Rewind and process
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract and accumulate flow in the box
        flow_box = flow[box_y:box_y+bh, box_x:box_x+bw]
        flow_sum += flow_box

        prev_gray = gray
        frame_count += 1

    # Now compute medians from summed flow
    total_dx = flow_sum[..., 0].flatten()
    total_dy = flow_sum[..., 1].flatten()

    avg_dx_per_col = np.mean(flow_sum[..., 0], axis=0)
    avg_dy_per_col = np.mean(flow_sum[..., 1], axis=0)

    median_dx = float(np.median(total_dx))
    '''
    Uncomment to plot per-column flow

    # Plotting both
    plt.figure()
    plt.plot(avg_dx_per_col, label='Average X Flow')
    plt.plot(avg_dy_per_col, label='Average Y Flow')
    plt.title("Average Optical Flow per Column")
    plt.xlabel("Column Index")
    plt.ylabel("Average Displacement")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    positive_dy = avg_dy_per_col[avg_dy_per_col > 0].flatten()
    median_dy = float(np.median(positive_dy))
    '''
    # Plot histogram
    plt.figure()
    plt.hist(positive_dy, bins=30)
    plt.title("Histogram of Positive Y Flow Values")
    plt.xlabel("Y Displacement")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    cap.release()
    '''

    # print(f"\n→ Overall Median dx = {median_dx:.2f}, dy = {median_dy:.2f}")
    return median_dx, median_dy



def process_dataset(dataset_filepath: str, original_dataset_filepath:str, output_dir: str, output_json: str, diff_keys = None):
    """Process entire dataset and save results"""
    # Load dataset
    dataset = load_json_data(dataset_filepath)
    original_dataset = load_json_data(original_dataset_filepath)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    output_data = {}
    flag = 1
    count = 0
    for sample_id, sample_data in tqdm(islice(dataset.items(), 50), desc="Processing samples", total=50):
        
        # Copy original data
        output_data[sample_id] = sample_data.copy()

        # Find corresponding annotation
        annotation_index = next(
            (i for i, ann in enumerate(original_dataset)
             if ann.get('s3_fileUrl') == sample_data['image_path']),
            None
        )

        if annotation_index is None:
            print(f"No annotation found for frame {sample_id}")
            continue

        # Process pedestrian annotations
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Pedestrian' and
            original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            output_data[sample_id]['Pedestrians'][str(len(sample_data['Pedestrians']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }

        # Process cyclist annotations
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Cyclist' and
            original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            output_data[sample_id]['Cyclists'][str(len(sample_data['Cyclists']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }
        try:
            # Get video dimensions from first frame
            video_path = download_video(output_data[sample_id]['video_path'])
            # animate_optical_flow(video_path)
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Initialize intent analyzer with more sensitive threshold
            intent_analyzer = IntentAnalyzer(frame_size=(width, height), motion_threshold=0.15)

            # Track objects in video and get their histories
            track_histories = process_video(output_data[sample_id]['video_path'], intent_analyzer)

            # Set camera motion in intent analyzer
            if 'camera' in track_histories:
                intent_analyzer.set_camera_motion(track_histories['camera']['centroids'])

            # Plot 3D tracks for this sample
            plot_3d_tracks(track_histories, f"Object Tracks - Sample {sample_id}")
            matched_tracks = {}
            # Process each object type separately
            for object_type in ['Pedestrians', 'Cyclists']:
                if object_type not in output_data[sample_id]:
                    continue

                # Get tracked objects of this type
                type_tracks = {k: v for k, v in track_histories.items()
                             if v['class'] == object_type}
                if not type_tracks:
                    continue

                # Get final boxes for matching
                track_ids = list(type_tracks.keys())
                tracked_boxes = [track_data['boxes'][-1] for track_data in type_tracks.values()]

                # Remove duplicate boxes from input data
                filtered_input_boxes = remove_duplicate_boxes(output_data[sample_id][object_type])

                # Match objects using Hungarian algorithm
                matches = match_objects(type_tracks, filtered_input_boxes)
                # Process matches
                for track_idx, obj_id in matches.items():
                    track_id = track_idx
                    track_data = type_tracks[track_id]
                    matched_tracks[track_id] = track_data

                    # Update track history for intent analysis
                    for i, centroid in enumerate(track_data['centroids']):
                        if i == 0:
                            cam_dx, cam_dy = get_median_optical_flow(video_path, (centroid[0], centroid[1]))

                            iter = 0
                            while cam_dy < 0:
                              cam_dx, cam_dy = get_median_optical_flow_multiple_margins(video_path, (centroid[0], centroid[1]), y_shift=True, margin_add=iter*5)
                              iter += 1
                              if iter > 10:
                                cam_dx, cam_dy = 0, 0
                                break
                        # Get camera motion

                        intent_analyzer.update_track_history(track_id, centroid)


                    # Analyze intent
                    intent = intent_analyzer.determine_intent(track_id, cam_dx, cam_dy)
                    position = intent_analyzer.determine_position(track_data['centroids'][-1])

                    # Update output data
                    output_data[sample_id][object_type][obj_id].update({
                        'Intent': intent,
                        'Position': position,
                        'Description': intent_analyzer.generate_description(intent)
                    })
            count += 1
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            continue
        if count % 100 == 0:
          print(f"Processed {count} samples")
          save_json_data(output_data, output_json)
    # Save results
    save_json_data(output_data, output_json)

import os
import cv2
from multiprocessing import Pool, cpu_count
from itertools import islice
from tqdm import tqdm

def process_dataset_parallel(dataset_filepath: str, original_dataset_filepath: str, output_dir: str, output_json: str):
    """Process entire dataset and save results"""
    # Load dataset
    dataset = load_json_data(dataset_filepath)
    original_dataset = load_json_data(original_dataset_filepath)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare items, skipping the first 550 just like `if count < 550: continue`
    items = list(islice(dataset.items(), 0, 5686))
    output_data = {}
    count = 0

    def worker(item):
        sample_id, sample_data = item

        

        # Copy original data
        local_output = { sample_id: sample_data.copy() }

        # Find corresponding annotation
        annotation_index = next(
            (i for i, ann in enumerate(original_dataset)
             if ann.get('s3_fileUrl') == sample_data['image_path']),
            None
        )
        if annotation_index is None:
            print(f"No annotation found for frame {sample_id}")
            return None

        # Process pedestrian annotations
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Pedestrian' and
            original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            local_output[sample_id]['Pedestrians'][str(len(sample_data['Pedestrians']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }

        # Process cyclist annotations
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Cyclist' and
            original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            local_output[sample_id]['Cyclists'][str(len(sample_data['Cyclists']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }

        try:
            # Get video dimensions from first frame
            video_path = download_video(local_output[sample_id]['video_path'])
            # animate_optical_flow(video_path)
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Initialize intent analyzer with more sensitive threshold
            intent_analyzer = IntentAnalyzer(frame_size=(width, height), motion_threshold=0.15)

            # Track objects in video and get their histories
            track_histories = process_video(local_output[sample_id]['video_path'], intent_analyzer)

            # Set camera motion in intent analyzer
            if 'camera' in track_histories:
                intent_analyzer.set_camera_motion(track_histories['camera']['centroids'])

            # Plot 3D tracks for this sample
            # plot_3d_tracks(track_histories, f"Object Tracks - Sample {sample_id}")

            # Process each object type separately
            for object_type in ['Pedestrians', 'Cyclists']:
                if object_type not in local_output[sample_id]:
                    continue

                # Get tracked objects of this type
                type_tracks = {k: v for k, v in track_histories.items()
                               if v['class'] == object_type}
                if not type_tracks:
                    continue

                # Get final boxes for matching
                track_ids = list(type_tracks.keys())
                tracked_boxes = [track_data['boxes'][-1] for track_data in type_tracks.values()]

                # Remove duplicate boxes from input data
                filtered_input_boxes = remove_duplicate_boxes(local_output[sample_id][object_type])

                # Match objects using Hungarian algorithm
                matches = match_objects(type_tracks, filtered_input_boxes)
                # Process matches
                for track_idx, obj_id in matches.items():
                    track_id = track_idx
                    track_data = type_tracks[track_id]

                    # Update track history for intent analysis
                    for i, centroid in enumerate(track_data['centroids']):
                        if i == 0:
                            cam_dx, cam_dy = get_median_optical_flow(video_path, (centroid[0], centroid[1]))

                            iter = 0
                            while cam_dy < 0:
                                cam_dx, cam_dy = get_median_optical_flow_multiple_margins(
                                    video_path,
                                    (centroid[0], centroid[1]),
                                    y_shift=True,
                                    margin_add=iter*5
                                )
                                iter += 1
                                if iter > 10:
                                    cam_dx, cam_dy = 0, 0
                                    break
                        # Get camera motion

                        intent_analyzer.update_track_history(track_id, centroid)

                    # Analyze intent
                    intent = intent_analyzer.determine_intent(track_id, cam_dx, cam_dy)
                    position = intent_analyzer.determine_position(track_data['centroids'][-1])

                    # Update output data
                    local_output[sample_id][object_type][obj_id].update({
                        'Intent': intent,
                        'Position': position,
                        'Description': intent_analyzer.generate_description(intent)
                    })

            return local_output

        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            print(sample_data)
            

    # Execute in parallel
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(worker, items),
                           total=len(items),
                           desc="Processing samples"):
            if result:
                output_data.update(result)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} samples")
                save_json_data(output_data, output_json)

    # Save final results
    save_json_data(output_data, output_json)



def box_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two boxes"""
    # Convert to x1,y1,x2,y2 format if needed
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter

    return inter / union if union > 0 else 0

def plot_3d_tracks(track_histories: Dict[int, Dict], title: str = "Object Tracks in 3D"):
    """
    Create a 3D plot of object tracks with time as the third dimension.

    Args:
        track_histories: Dictionary mapping track_id to track data containing centroids
        title: Title for the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # First plot camera motion if available
    if 'camera' in track_histories:
        camera_data = track_histories.pop('camera')
        camera_points = np.array(camera_data['centroids'])
        if len(camera_points) > 1:
            ax.plot(camera_points[:, 0], camera_points[:, 1],
                   camera_data['frame_nums'],  # Use actual frame numbers
                   c='red', linewidth=2, label='Camera Motion',
                   linestyle='--')

    # Then plot object tracks
    colors = plt.cm.rainbow(np.linspace(0, 1, len(track_histories)))

    for (track_id, track_data), color in zip(track_histories.items(), colors):
        centroids = track_data.get('centroids', [])
        frame_nums = track_data.get('frame_nums', [])  # Get frame numbers
        if len(centroids) < 2:
            continue

        # Convert centroids to numpy arrays
        points = np.array(centroids)
        xs = points[:, 0]
        ys = points[:, 1]

        # Plot the track
        ax.plot(xs, ys, frame_nums, c=color,
               label=f'Track {track_id} ({track_data["class"]}) [F{frame_nums[0]}-{frame_nums[-1]}]')
        # Plot start point
        ax.scatter(xs[0], ys[0], frame_nums[0], c=color, marker='o')
        # Plot end point
        ax.scatter(xs[-1], ys[-1], frame_nums[-1], c=color, marker='^')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Frame Number')
    # ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Track persons and analyze their intent in videos')
    parser.add_argument('--dataset_filepath', type=str, required=True,
                      help='Path to the input JSON file containing video URLs')
    parser.add_argument('--original_dataset_filepath', type=str, required=True,
                      help='Path to the original DRAMA dataset JSON file containing video URLs')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory for output videos')
    parser.add_argument('--output_json', type=str, required=True,
                      help='Path for output JSON with tracking and intent data')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold for detections')
    parser.add_argument('--parallel', type=bool, default=False,
                      help='True for multithreading')

    args = parser.parse_args()
    if args.parallel:
      process_dataset_parallel(args.dataset_filepath, args.original_dataset_filepath, args.output_dir, args.output_json)
    else:
      process_dataset(args.dataset_filepath, args.original_dataset_filepath, args.output_dir, args.output_json )

if __name__ == "__main__":
    main()