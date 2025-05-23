import json
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm
import logging
import sys
model = YOLO('yolov8n.pt')

def calculate_iou(box1, box2):
    """
    Calculate intersection over union between two boxes
    Each box should be in format [x1, y1, x2, y2]
    """
    # Get coordinates of box1
    x1_1, y1_1, x2_1, y2_1 = box1
    # Get coordinates of box2
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Calculate areas
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def check_visibility(box, image_width, image_height, min_visibility=0.5):
    """
    Check if object is sufficiently visible in the scene
    Returns True if the object is visible above the threshold
    """
    x1, y1, x2, y2 = box

    # Calculate box dimensions
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height

    # Calculate how much of the box is outside the image
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(image_width, x2)
    y2_clip = min(image_height, y2)

    # Calculate visible area
    visible_width = x2_clip - x1_clip
    visible_height = y2_clip - y1_clip
    visible_area = visible_width * visible_height

    # Calculate visibility ratio
    visibility_ratio = visible_area / box_area

    return visibility_ratio >= min_visibility, visibility_ratio

def check_size(box, image_width, image_height, min_height_ratio=0.1, min_width_ratio=0.05):
    """
    Check if object meets minimum size requirements relative to image dimensions
    Returns True if the object is large enough, and the actual ratios
    """
    x1, y1, x2, y2 = box

    # Calculate box dimensions
    box_width = x2 - x1
    box_height = y2 - y1

    # Calculate size ratios
    height_ratio = box_height / image_height
    width_ratio = box_width / image_width

    # Calculate area for sorting
    area = box_width * box_height

    meets_threshold = (height_ratio >= min_height_ratio and width_ratio >= min_width_ratio)
    return meets_threshold, height_ratio, width_ratio, area

def filter_and_sort_detections(results, image_width, image_height,
                             min_visibility=0.5, conf_threshold=0.3,
                             min_height_ratio=0.1, min_width_ratio=0.05,
                             max_per_class=3):
    """
    Filter detections based on all criteria and return the largest ones
    """
    filtered_detections = {
        'person': [],
        'cyclist': []
    }

    # First, filter and collect all valid detections
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)

            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Convert to numpy array

            # Check visibility
            vis_pass, vis_ratio = check_visibility(
                [x1, y1, x2, y2], image_width, image_height, min_visibility
            )

            # Check size
            size_pass, height_ratio, width_ratio, area = check_size(
                [x1, y1, x2, y2], image_width, image_height,
                min_height_ratio, min_width_ratio
            )

            if vis_pass and size_pass:
                if cls == 'person':
                    filtered_detections['person'].append({
                        'box': [x1, y1, x2, y2],  # Store coordinates instead of box object
                        'area': area,
                        'conf': conf,
                        'vis_ratio': vis_ratio,
                        'height_ratio': height_ratio,
                        'width_ratio': width_ratio
                    })
                elif cls == 'bicycle':
                    # Check if person is on bicycle
                    for person_det in filtered_detections['person']:
                        person_box = person_det['box']
                        if calculate_iou(person_box, [x1, y1, x2, y2]) > 0.5:
                            filtered_detections['cyclist'].append({
                                'person_box': person_box,
                                'bicycle_box': [x1, y1, x2, y2],
                                'area': area,
                                'conf': conf,
                                'vis_ratio': vis_ratio,
                                'height_ratio': height_ratio,
                                'width_ratio': width_ratio
                            })
                            break

    # Sort by area and keep only the largest ones
    for cls in filtered_detections:
        filtered_detections[cls].sort(key=lambda x: x['area'], reverse=True)
        filtered_detections[cls] = filtered_detections[cls][:max_per_class]

    return filtered_detections

def plot_boxes(image, results, min_visibility=0.5, conf_threshold=0.3,
               min_height_ratio=0.1, min_width_ratio=0.05, max_per_class=3):
    """
    Plot bounding boxes on the image with all filtering criteria
    """
    # Convert PIL Image to numpy array
    img = np.array(image)
    image_width, image_height = image.size

    # Create figure and axes
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Define colors for different classes
    colors = {
        'person': 'red',
        'cyclist': 'blue'
    }

    # Get filtered detections
    filtered_detections = filter_and_sort_detections(
        results, image_width, image_height,
        min_visibility, conf_threshold,
        min_height_ratio, min_width_ratio,
        max_per_class
    )

    # Plot each class
    for cls, detections in filtered_detections.items():
        for i, det in enumerate(detections):
            if cls == 'person':
                x1, y1, x2, y2 = det['box']
            elif cls == 'cyclist':
                x1, y1 = det['person_box'][:2]
                x2, y2 = det['bicycle_box'][2:]

            # Create rectangle patch
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                    color=colors[cls], linewidth=2)

            # Add label with all metrics
            plt.text(x1, y1-5,
                    f'{cls} #{i+1}\nconf:{det["conf"]:.2f}\n',
                    color=colors[cls], fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')
    plt.show()


def count_objects(image_url, min_visibility=0.5, conf_threshold=0.3,
                 min_height_ratio=0.1, min_width_ratio=0.05, max_per_class=3):
    # try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))

        # Perform object detection
        results = model(image)

        # Plot the image with bounding boxes
        # plot_boxes(image, results, min_visibility, conf_threshold,
        #           min_height_ratio, min_width_ratio, max_per_class)

        # Get filtered detections
        filtered_detections = filter_and_sort_detections(
            results, image.size[0], image.size[1],
            min_visibility, conf_threshold,
            min_height_ratio, min_width_ratio,
            max_per_class
        )

        # Count objects (will be maximum max_per_class per class)
        counts = {
            "Pedestrians": len(filtered_detections['person']),
            "Cyclists": len(filtered_detections['cyclist'])
        }

        return counts


def is_person_on_bicycle(person_box, bicycle_box, iou_threshold=0.3):
    """
    Check if a person is positioned correctly on a bicycle:
    1. Person's bottom should align with bicycle's top half
    2. IOU should be above threshold
    3. Person should be vertically above the bicycle
    """
    # Unpack coordinates
    person_x1, person_y1, person_x2, person_y2 = person_box
    bicycle_x1, bicycle_y1, bicycle_x2, bicycle_y2 = bicycle_box

    # Check IOU
    iou = calculate_iou(person_box, bicycle_box)
    if iou < iou_threshold:
        return False

    # Check vertical positioning
    bicycle_height = bicycle_y2 - bicycle_y1
    bicycle_midpoint = bicycle_y1 + (bicycle_height / 2)

    # Person's bottom (y2) should be around the middle of the bicycle
    # and person's top (y1) should be above bicycle's top
    if (person_y2 >= bicycle_y1 and
        person_y2 <= bicycle_y2 and
        person_y1 < bicycle_y1):
        return True

    return False

def count_raw_detections(results, image_width, image_height, conf_threshold=0.3):
    """
    Count all detections before applying size and visibility filters
    """
    person_count = 0
    potential_cyclists = 0

    for r in results:
        # First count all persons
        person_boxes = []
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)

            if conf < conf_threshold:
                continue

            if cls == 'person':
                person_count += 1
                person_boxes.append(box.xyxy[0].cpu().numpy())

        # Then check for bicycles with properly positioned persons
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)

            if conf < conf_threshold or cls != 'bicycle':
                continue

            bicycle_box = box.xyxy[0].cpu().numpy()
            # Check each person for proper positioning with this bicycle
            for person_box in person_boxes:
                if is_person_on_bicycle(person_box, bicycle_box):
                    potential_cyclists += 1
                    break  # One bicycle can only have one cyclist

    return person_count, potential_cyclists

def filter_and_sort_detections(results, image_width, image_height,
                             min_visibility=0.5, conf_threshold=0.3,
                             min_height_ratio=0.1, min_width_ratio=0.05,
                             max_per_class=3):
    """
    Filter detections based on all criteria and return the largest ones
    """
    filtered_detections = {
        'person': [],
        'cyclist': []
    }

    # First, collect all valid person detections
    person_detections = []
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)

            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Check visibility
            vis_pass, vis_ratio = check_visibility(
                [x1, y1, x2, y2], image_width, image_height, min_visibility
            )

            # Check size
            size_pass, height_ratio, width_ratio, area = check_size(
                [x1, y1, x2, y2], image_width, image_height,
                min_height_ratio, min_width_ratio
            )

            if vis_pass and size_pass:
                if cls == 'person':
                    person_detections.append({
                        'box': [x1, y1, x2, y2],
                        'area': area,
                        'conf': conf,
                        'vis_ratio': vis_ratio,
                        'height_ratio': height_ratio,
                        'width_ratio': width_ratio
                    })

    # Then look for bicycles and check for proper cyclist positioning
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)

            if conf < conf_threshold or cls != 'bicycle':
                continue

            bicycle_box = box.xyxy[0].cpu().numpy()

            # Check visibility and size for bicycle
            vis_pass, vis_ratio = check_visibility(
                bicycle_box, image_width, image_height, min_visibility
            )

            size_pass, height_ratio, width_ratio, area = check_size(
                bicycle_box, image_width, image_height,
                min_height_ratio, min_width_ratio
            )

            if not (vis_pass and size_pass):
                continue

            # Check each person for proper positioning with this bicycle
            for person_det in person_detections:
                if is_person_on_bicycle(person_det['box'], bicycle_box):
                    filtered_detections['cyclist'].append({
                        'person_box': person_det['box'],
                        'bicycle_box': bicycle_box,
                        'area': max(area, person_det['area']),  # Use larger of the two areas
                        'conf': min(conf, person_det['conf']),  # Use lower of the two confidences
                        'vis_ratio': min(vis_ratio, person_det['vis_ratio']),
                        'height_ratio': max(height_ratio, person_det['height_ratio']),
                        'width_ratio': max(width_ratio, person_det['width_ratio'])
                    })
                    # Remove this person from potential pedestrians
                    if person_det in person_detections:
                        person_detections.remove(person_det)
                    break  # One bicycle can only have one cyclist

    # Add remaining persons as pedestrians
    filtered_detections['person'] = person_detections

    # Sort by area and keep only the largest ones
    for cls in filtered_detections:
        filtered_detections[cls].sort(key=lambda x: x['area'], reverse=True)
        filtered_detections[cls] = filtered_detections[cls][:max_per_class]

    return filtered_detections


def save_annotated_image(image, results, filtered_detections, save_path, min_visibility=0.5,
                        conf_threshold=0.3, min_height_ratio=0.1, min_width_ratio=0.05):
    """
    Save image with bounding boxes but without labels
    """
    # Convert PIL Image to numpy array
    img = np.array(image)

    # Create figure and axes
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Define colors for different classes
    colors = {
        'person': 'red',
        'cyclist': 'blue'
    }

    # Plot each class
    for cls, detections in filtered_detections.items():
        for det in detections:
            if cls == 'person':
                x1, y1, x2, y2 = det['box']
            elif cls == 'cyclist':
                x1, y1 = det['person_box'][:2]
                x2, y2 = det['bicycle_box'][2:]

            # Create rectangle patch
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                    color=colors[cls], linewidth=0.7)

    plt.axis('off')

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def create_directory_structure(base_dir, image_url):
    """
    Create directory structure based on image URL
    Returns the complete save path for the image
    """
    try:
        # Extract the path after 'drama/combined/'
        path_parts = image_url.split('drama/combined/')
        if len(path_parts) != 2:
            return None

        # Get the relative path (e.g., 'titan/clip_305_000786/frame_000786.png')
        relative_path = path_parts[1]

        # Split into directory path and filename
        dir_path = os.path.dirname(relative_path)  # titan/clip_305_000786
        filename = os.path.basename(relative_path)  # frame_000786.png

        # Create complete directory path
        full_dir_path = os.path.join(base_dir, dir_path)

        # Create all necessary directories
        os.makedirs(full_dir_path, exist_ok=True)

        # Return complete save path for the image
        return os.path.join(full_dir_path, filename), filename, dir_path
    except Exception as e:
        logging.info(f"Error creating directory structure for {image_url}: {e}")
        return None, None, None

def save_annotated_image(image, results, filtered_detections, save_path):
    """
    Save image with bounding boxes but without labels
    """
    try:
        # Convert PIL Image to numpy array
        img = np.array(image)

        # Create figure and axes
        plt.figure(figsize=(12, 8))
        plt.imshow(img)

        # Define colors for different classes
        colors = {
            'person': 'red',
            'cyclist': 'blue'
        }

        # Plot each class
        for cls, detections in filtered_detections.items():
            for det in detections:
                if cls == 'person':
                    x1, y1, x2, y2 = det['box']
                elif cls == 'cyclist':
                    x1, y1 = det['person_box'][:2]
                    x2, y2 = det['bicycle_box'][2:]

                # Create rectangle patch
                plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                        color=colors[cls], linewidth=2)

        plt.axis('off')

        # Save the figure
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        return True
    except Exception as e:
        logging.info(f"Error saving annotated image to {save_path}: {e}")
        return False

def process_and_save_data(input_json_path, base_output_dir, base_json_dir, max_total_detections=3):
    """
    Process images and save results in the specified format
    """
    try:
        # Load YOLO model
        model = YOLO('yolov8n.pt')

        # Create base directory if it doesn't exist
        os.makedirs(base_output_dir, exist_ok=True)

        # Load input data
        with open(input_json_path, 'r') as f:
            data = json.load(f)

        # Initialize output JSON
        output_json = {}
        # Process each image
        for item in tqdm(data):
            image_url = item.get("s3_fileUrl")
            if not image_url:
                continue

            # Create directory structure and get save path
            save_path, filename, dir_path = create_directory_structure(base_output_dir, image_url)
            if not save_path:
                logging.info(f"Skipping {image_url} due to invalid path structure")
                continue

            logging.info(f"Processing image: {filename}")
            logging.info(f"Save path: {save_path}")
            logging.info(f"Directory path: {dir_path}")

            try:
                # Download and process image
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))

                # Perform detection
                results = model(image, verbose=False)

                # Get raw counts
                raw_person_count, raw_cyclist_count = count_raw_detections(
                    results, image.size[0], image.size[1], conf_threshold=0.5
                )

                # Skip if too many detections
                if raw_person_count > max_total_detections or raw_cyclist_count > max_total_detections:
                    logging.info(f"Skipping {filename} - too many detections")
                    continue

                # Get filtered detections
                filtered_detections = filter_and_sort_detections(
                    results, image.size[0], image.size[1],
                    min_visibility=0.5, conf_threshold=0.5,
                    min_height_ratio=0.08, min_width_ratio=0.01,
                    max_per_class=3
                )

                # Skip if no detections
                if not (filtered_detections['person'] or filtered_detections['cyclist']):
                    logging.info(f"Skipping {filename} - no valid detections")
                    continue

                # Save annotated image
                if not save_annotated_image(image, results, filtered_detections, save_path):
                    continue

                # Prepare JSON entry
                json_entry = {
                    "image_path": image_url,
                    "video_path": item.get("s3_instructionReference", ""),
                    "Risk": item.get("Risk", ""),
                    "Pedestrians": {},
                    "Cyclists": {}
                }

                # Add pedestrian detections
                for idx, det in enumerate(filtered_detections['person'], 1):
                    json_entry["Pedestrians"][str(idx)] = {
                        "Box": [int(x) for x in det['box']],
                        "Intent": ""
                    }

                # Add cyclist detections
                for idx, det in enumerate(filtered_detections['cyclist'], 1):
                    print('Yayyyyyy')
                    # For cyclists, use the combined box
                    x1 = min(int(det['person_box'][0]), int(det['bicycle_box'][0]))
                    y1 = min(int(det['person_box'][1]), int(det['bicycle_box'][1]))
                    x2 = max(int(det['person_box'][2]), int(det['bicycle_box'][2]))
                    y2 = max(int(det['person_box'][3]), int(det['bicycle_box'][3]))

                    json_entry["Cyclists"][str(idx)] = {
                        "Box": [x1, y1, x2, y2],
                        "Intent": ""
                    }

                # Add to output JSON
                output_json[f'{dir_path.split("/")[-1]}_{filename.split(".")[0]}'] = json_entry
                logging.info(f"Successfully processed {filename}")

            except Exception as e:
                logging.info(f"Error processing {filename}: {e}")
                continue
        # Save JSON file
        print(output_json)
        json_output_path = os.path.join(base_json_dir, 'drama_intent_temp.json')
        with open(json_output_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        f.close()
        logging.info(f"Successfully saved drama_intent.json")

    except Exception as e:
        logging.info(f"Fatal error in process_and_save_data: {e}")

def setup_logging(log_file='/content/drive/MyDrive/GenAI/DRAMA/processing.log'):
    """Setup logging configuration"""
    logging.basicConfig(
            level=logging.INFO,
            force=True,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file)  # Logs only to file
            ]
        )

# Main execution
if __name__ == "__main__":
    input_json_path = '<path-to>/integrated_output_v2.json'
    base_output_dir = '<output-dir>'
    base_json_dir = '<json-save-dir>'
    setup_logging()

    # Start processing
    logging.info("Starting data processing...")
    process_and_save_data(input_json_path, base_output_dir, base_json_dir)