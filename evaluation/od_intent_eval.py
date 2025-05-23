import json
import os
import numpy as np
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
import requests
from PIL import Image
import io
import base64
from io import BytesIO
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2: Lists of [x1, y1, x2, y2] coordinates

    Returns:
    float: IoU value
    """
    # Convert to [x1, y1, x2, y2] format if needed
    if len(box1) == 4 and len(box2) == 4:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou
    else:
        print(f"Error: Invalid box format. Expected 4 values, got {len(box1)} and {len(box2)}")
        # raise
        return 0.0

def is_same_object_class(gt_class, pred_class):
    """
    Check if the ground truth and predicted object classes are similar.

    Parameters:
    gt_class: Ground truth class name
    pred_class: Predicted class name

    Returns:
    bool: True if classes match, False otherwise
    """
    # Convert to lowercase for case-insensitive comparison
    gt_class = gt_class.lower()
    pred_class = pred_class.lower()

    # Define similar class names
    pedestrian_classes = ['pedestrian', 'person', 'human', 'walker']
    cyclist_classes = ['cyclist', 'bike', 'bicycle', 'bicyclist']

    # Check if both classes belong to the same category
    if gt_class in pedestrian_classes and pred_class in pedestrian_classes:
        return True
    if gt_class in cyclist_classes and pred_class in cyclist_classes:
        return True

    # Direct match
    return gt_class == pred_class
def image_dimensions(image_url,ind):
    # response = requests.get(image_url, stream=True)
    # response.raise_for_status()
    # image = Image.open(io.BytesIO(response.content))
    # imagenp = np.array(image)
    # print(imagenp.shape)
    shape = (1520, 2704)
    return shape[1] if ind % 2 == 1 else shape[0]
    return imagenp.shape[0] if ind % 2 == 0 else imagenp.shape[1]

def evaluate_bounding_boxes(ground_truth, predictions,num_samples, iou_threshold=0.25):
    """
    Evaluate bounding box detection accuracy.

    Parameters:
    ground_truth: Dictionary containing ground truth annotations
    predictions: Dictionary containing predicted bounding boxes
    iou_threshold: IoU threshold for considering a detection as correct

    Returns:
    dict: Dictionary containing evaluation metrics
    """
    total_gt_objects = 0
    total_correct_detections = 0
    class_metrics = defaultdict(lambda: {'total': 0, 'correct': 0})

    for sample_id, gt_sample in tqdm(islice(ground_truth.items(), num_samples), total = num_samples):
        try:
          if sample_id not in predictions:
              continue

          pred_sample = predictions[sample_id]

          # Process pedestrians
          if 'Pedestrians' in gt_sample:
              for ped_id, ped_data in gt_sample['Pedestrians'].items():
                  total_gt_objects += 1
                  class_metrics['pedestrian']['total'] += 1

                  gt_box = ped_data['Box']
                  best_iou = 0
                  best_match = None
                  if isinstance(pred_sample, list):
                    pred_sample = pred_sample[0]
                  # Find the best matching prediction
                  for pred_obj, pred_det in pred_sample.items():
                      if 'pedestrian' in pred_obj.lower() or 'person' in pred_obj.lower():
                          if 'Bounding_box' in pred_det or 'bounding_box' in pred_det:
                            pred_box = pred_det['Bounding_box'][0] if len(pred_det['Bounding_box']) == 1 else pred_det['Bounding_box']
                            if not pred_box or isinstance(pred_box, int):
                              continue
                            if isinstance(pred_box, dict):
                                  pred_box = list(pred_box.values())
                            if isinstance(pred_box[0], list):
                              for i, box in enumerate(pred_box):
                                iou = calculate_iou(gt_box, pred_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = pred_det
                            else:
                              pred_box = [0 if v is None else float(v) for v in pred_box]
                              pred_box = [v*image_dimensions(gt_sample['image_path'], i) if 0 < v < 1 else v for i, v in enumerate(pred_box)]
                              iou = calculate_iou(gt_box, pred_box)

                              if iou > best_iou:
                                  best_iou = iou
                                  best_match = pred_det

                  # Check if the best match exceeds the IoU threshold
                  if best_match and best_iou >= iou_threshold:
                      total_correct_detections += 1
                      class_metrics['pedestrian']['correct'] += 1

          # Process cyclists
          if 'Cyclists' in gt_sample:
              for cyc_id, cyc_data in gt_sample['Cyclists'].items():
                  total_gt_objects += 1
                  class_metrics['cyclist']['total'] += 1

                  gt_box = cyc_data['Box']
                  best_iou = 0
                  best_match = None

                  # Find the best matching prediction
                  for pred_obj, pred_det in pred_sample.items():
                      if 'cyclist' in pred_obj.lower():
                          if 'Bounding_box' in pred_det or 'bounding_box' in pred_det:
                            pred_box = pred_det['Bounding_box'][0] if len(pred_det['Bounding_box']) == 1 else pred_det['Bounding_box']
                            if not pred_box or isinstance(pred_box, int):
                              continue
                            if isinstance(pred_box, dict):
                                  pred_box = list(pred_box.values())
                            if isinstance(pred_box[0], list):
                              for i, box in enumerate(pred_box):
                                iou = calculate_iou(gt_box, pred_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = pred_det
                            else:
                              pred_box = [0 if v is None else float(v) for v in pred_box]
                              pred_box = [v*image_dimensions(gt_sample['image_path'], i) if 0 < v < 1 else v for i, v in enumerate(pred_box)]
                              iou = calculate_iou(gt_box, pred_box)

                              if iou > best_iou:
                                  best_iou = iou
                                  best_match = pred_det

                  # Check if the best match exceeds the IoU threshold
                  if best_match and best_iou >= iou_threshold:
                      total_correct_detections += 1
                      class_metrics['cyclist']['correct'] += 1
        except:
          print(f'Error in sample {sample_id}, {pred_box}')
          # raise
          continue

    # Calculate overall accuracy
    overall_accuracy = total_correct_detections / total_gt_objects if total_gt_objects > 0 else 0

    # Calculate per-class accuracy
    class_accuracies = {}
    for class_name, metrics in class_metrics.items():
        class_accuracies[class_name] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0

    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'total_objects': total_gt_objects,
        'correct_detections': total_correct_detections
    }

def evaluate_intents(ground_truth, predictions, num_samples):
    """
    Evaluate intent prediction accuracy.

    Parameters:
    ground_truth: Dictionary containing ground truth annotations
    predictions: Dictionary containing predicted intents

    Returns:
    dict: Dictionary containing evaluation metrics
    """
    total_gt_intents = 0
    total_correct_intents = 0
    total_horizontal_intents = 0
    total_vertical_intents = 0
    correct_horizontal_intents = 0
    correct_vertical_intents = 0

    for sample_id, gt_sample in tqdm(islice(ground_truth.items(), num_samples), total = num_samples):
        try:
          # Process pedestrians
          if 'Pedestrians' in gt_sample:
              for ped_id, ped_data in gt_sample['Pedestrians'].items():
                  if 'Intent' not in ped_data or not ped_data['Intent']:
                      continue

                  gt_intent = ped_data['Intent']

                  # Skip if intent is not a list or is empty
                  if not isinstance(gt_intent, list) or len(gt_intent) == 0:
                      continue

                  # Count all ground truth intents
                  if len(gt_intent) > 0:
                      total_horizontal_intents += 1
                  if len(gt_intent) > 1:
                      total_vertical_intents += 1

                  total_gt_intents += len(gt_intent)

                  # Only try to match with predictions if the sample exists in predictions
                  if sample_id in predictions:
                      pred_sample = predictions[sample_id]
                      gt_box = ped_data['Box']

                      best_iou = 0
                      best_match = None

                      # Find the best matching prediction based on IoU
                      for pred_obj, pred_det in pred_sample.items():
                          if 'pedestrian' in pred_obj.lower() or 'person' in pred_obj.lower():
                              if 'Bounding_box' in pred_det or 'bounding_box' in pred_det:
                                pred_box = pred_det['Bounding_box'][0] if len(pred_det['Bounding_box']) == 1 else pred_det['Bounding_box']
                                if not pred_box or isinstance(pred_box, int):
                                    continue
                                if isinstance(pred_box, dict):
                                  pred_box = list(pred_box.values())
                                if isinstance(pred_box[0], list):
                                  for i, box in enumerate(pred_box):
                                    iou = calculate_iou(gt_box, pred_box)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_match = pred_det
                                else:
                                  pred_box = [0 if v is None else float(v) for v in pred_box]
                                  pred_box = [v*image_dimensions(gt_sample['image_path'], i) if 0 < v < 1 else v for i, v in enumerate(pred_box)]
                                  iou = calculate_iou(gt_box, pred_box)

                                  if iou > best_iou:
                                      best_iou = iou
                                      best_match = pred_det

                      # Check if we found a match with sufficient IoU
                      if best_match and best_iou >= 0.6:
                          pred_intent = best_match.get('Intent', [])
                          print(pred_intent)
                          if isinstance(pred_intent, str):
                            pred_intent = [pred_intent, pred_intent]
                          # print(pred_intent
                          # Compare horizontal intent (first element)
                          if len(gt_intent) > 0 and len(pred_intent) > 0:
                              if gt_intent[0] == pred_intent[0]:
                                  correct_horizontal_intents += 1
                                  total_correct_intents += 1

                          # Compare vertical intent (second element)
                          if len(gt_intent) > 1 and len(pred_intent) > 1:
                              if gt_intent[1] == pred_intent[1]:
                                  correct_vertical_intents += 1
                                  total_correct_intents += 1

          # Process cyclists
          if 'Cyclists' in gt_sample:
              for cyc_id, cyc_data in gt_sample['Cyclists'].items():
                  if 'Intent' not in cyc_data or not cyc_data['Intent']:
                      continue

                  gt_intent = cyc_data['Intent']

                  # Skip if intent is not a list or is empty
                  if not isinstance(gt_intent, list) or len(gt_intent) == 0:
                      continue

                  # Count all ground truth intents
                  if len(gt_intent) > 0:
                      total_horizontal_intents += 1
                  if len(gt_intent) > 1:
                      total_vertical_intents += 1

                  total_gt_intents += len(gt_intent)

                  # Only try to match with predictions if the sample exists in predictions
                  if sample_id in predictions:
                      pred_sample = predictions[sample_id]
                      gt_box = cyc_data['Box']

                      best_iou = 0
                      best_match = None

                      # Find the best matching prediction based on IoU
                      for pred_obj, pred_det in pred_sample.items():
                          if 'cyclist' in pred_obj.lower():
                              if 'Bounding_box' in pred_det or 'bounding_box' in pred_det:
                                pred_box = pred_det['Bounding_box'][0] if len(pred_det['Bounding_box']) == 1 else pred_det['Bounding_box']
                                pred_box = [v*image_dimensions(gt_sample['image_path'], i) if 0 < v < 1 else v for i, v in enumerate(pred_box)]
                                if not pred_box or isinstance(pred_box, int):
                                    continue
                                if isinstance(pred_box, dict):
                                  pred_box = list(pred_box.values())
                                if isinstance(pred_box[0], list):
                                  for i, box in enumerate(pred_box):
                                    iou = calculate_iou(gt_box, pred_box)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_match = pred_det
                                else:
                                  pred_box = [0 if v is None else float(v) for v in pred_box]
                                  pred_box = [v*image_dimensions(gt_sample['image_path'], i) if 0 < v < 1 else v for i, v in enumerate(pred_box)]
                                  iou = calculate_iou(gt_box, pred_box)

                                  if iou > best_iou:
                                      best_iou = iou
                                      best_match = pred_det

                      # Check if we found a match with sufficient IoU
                      if best_match and best_iou >= 0.6:
                          pred_intent = best_match.get('Intent', [])

                          # Compare horizontal intent (first element)
                          if len(gt_intent) > 0 and len(pred_intent) > 0:
                              if gt_intent[0] == pred_intent[0]:
                                  correct_horizontal_intents += 1
                                  total_correct_intents += 1

                          # Compare vertical intent (second element)
                          if len(gt_intent) > 1 and len(pred_intent) > 1:
                              if gt_intent[1] == pred_intent[1]:
                                  correct_vertical_intents += 1
                                  total_correct_intents += 1
        except:
          print(f'Error in sample {sample_id}')
          # raise
          continue
    # Calculate overall intent accuracy
    overall_accuracy = total_correct_intents / total_gt_intents if total_gt_intents > 0 else 0

    # Calculate per-direction intent accuracy
    horizontal_accuracy = correct_horizontal_intents / total_horizontal_intents if total_horizontal_intents > 0 else 0
    vertical_accuracy = correct_vertical_intents / total_vertical_intents if total_vertical_intents > 0 else 0

    return {
        'overall_accuracy': overall_accuracy,
        'horizontal_accuracy': horizontal_accuracy,
        'vertical_accuracy': vertical_accuracy,
        'total_intents': total_gt_intents,
        'correct_intents': total_correct_intents,
        'total_horizontal': total_horizontal_intents,
        'correct_horizontal': correct_horizontal_intents,
        'total_vertical': total_vertical_intents,
        'correct_vertical': correct_vertical_intents
    }

def main():
    # File paths
    ground_truth_path = '<dataset-path>'
    predictions_path = '<predictions-path>'
    # Load ground truth data
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # Load predictions
    try:
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"Predictions file not found: {predictions_path}")
        print("Creating a sample predictions file structure for demonstration...")

        # Create a sample predictions structure for demonstration
        predictions = {}
        for sample_id, sample_data in islice(ground_truth.items(), 1):
            predictions[sample_id] = []

            # Process pedestrians
            if 'Pedestrians' in sample_data:
                for ped_id, ped_data in sample_data['Pedestrians'].items():
                    if 'Box' in ped_data and 'Intent' in ped_data and isinstance(ped_data['Intent'], list):
                        # Create a sample prediction with slight box variation
                        box = ped_data['Box']
                        pred_obj = {
                            'name_of_object': f'Pedestrian{ped_id}',
                            'Intent': ped_data['Intent'][:],  # Copy the intent list
                            'Reason': f"Prediction reason for pedestrian {ped_id}",
                            'Bounding_box': [
                                box[0] + np.random.randint(-20, 20),
                                box[1] + np.random.randint(-20, 20),
                                box[2] + np.random.randint(-20, 20),
                                box[3] + np.random.randint(-20, 20)
                            ]
                        }
                        predictions[sample_id].append(pred_obj)

            # Process cyclists
            if 'Cyclists' in sample_data:
                for cyc_id, cyc_data in sample_data['Cyclists'].items():
                    if 'Box' in cyc_data and 'Intent' in cyc_data and isinstance(cyc_data['Intent'], list):
                        # Create a sample prediction with slight box variation
                        box = cyc_data['Box']
                        pred_obj = {
                            'name_of_object': f'Cyclist{cyc_id}',
                            'Intent': cyc_data['Intent'][:],  # Copy the intent list
                            'Reason': f"Prediction reason for cyclist {cyc_id}",
                            'Bounding_box': [
                                box[0] + np.random.randint(-20, 20),
                                box[1] + np.random.randint(-20, 20),
                                box[2] + np.random.randint(-20, 20),
                                box[3] + np.random.randint(-20, 20)
                            ]
                        }
                        predictions[sample_id].append(pred_obj)

        # Save the sample predictions
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"Created sample predictions file: {predictions_path}")

    # Evaluate bounding boxes
    bbox_results = evaluate_bounding_boxes(ground_truth, predictions, num_samples = len(predictions))

    # Evaluate intents
    intent_results = evaluate_intents(ground_truth, predictions,  num_samples = len(predictions))

    # Print results
    print("\n===== Bounding Box Evaluation =====")
    print(f"Overall Accuracy: {bbox_results['overall_accuracy']:.4f}")
    print("Per-Class Accuracy:")
    for class_name, accuracy in bbox_results['class_accuracies'].items():
        print(f"  {class_name}: {accuracy:.4f}")
    print(f"Total Objects: {bbox_results['total_objects']}")
    print(f"Correct Detections: {bbox_results['correct_detections']}")

    print("\n===== Intent Evaluation =====")
    print(f"Overall Accuracy: {intent_results['overall_accuracy']:.4f}")
    print(f"Horizontal Intent Accuracy: {intent_results['horizontal_accuracy']:.4f}")
    print(f"Vertical Intent Accuracy: {intent_results['vertical_accuracy']:.4f}")
    print(f"Total Intents: {intent_results['total_intents']}")
    print(f"Correct Intents: {intent_results['correct_intents']}")

    # Save results to a file
    results = {
        'bounding_box_evaluation': bbox_results,
        'intent_evaluation': intent_results
    }

    with open('<result-json-path>', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation results saved to <result-json-path>")

if __name__ == "__main__":
    main()
