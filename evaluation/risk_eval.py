import os
import json
import logging
from tqdm import tqdm
from itertools import islice
import openai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import random
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


openai.api_key  = '<api-key>'

def generate_response(prompt, api_key=None):
    """
    Generate a response using OpenAI API

    Parameters:
    prompt: The prompt to send to the API
    api_key: Optional API key (if not set in environment)

    Returns:
    str: The generated response
    """
    try:
        
        response =  openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes traffic scenes for risk assessment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0,
        )

        response_text = response.choices[0].message.content
        return response_text.strip()
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        return None

def evaluate_risk(ground_truth, intent_predictions, scene_graph_dir=None, num_samples=10, api_key=None, raw = False, raw_risk_predictions = None, gt = False):
    """
    Evaluate risk prediction accuracy

    Parameters:
    ground_truth: Dictionary containing ground truth annotations with Risk field
    intent_predictions: Dictionary containing predicted intents
    scene_graph_dir: Directory containing scene graph files (optional)
    num_samples: Number of samples to evaluate
    api_key: OpenAI API key

    Returns:
    dict: Dictionary containing evaluation metrics
    """
    true_risks = []
    pred_risks = []
    processed_samples = 0
    risk_predictions = {}  # Dictionary to store risk predictions

    for sample_id, gt_sample in tqdm(islice(ground_truth.items(), num_samples), total=num_samples, desc="Evaluating risk"):

        try:
            # Get ground truth risk
            true_risk = gt_sample.get('Risk', '')
            if not true_risk:
                logger.warning(f"No risk information for sample {sample_id}")
                continue

            # Standardize risk values to Yes/No
            true_risk = 'Yes' if true_risk.lower() == 'yes' else 'No'
            true_risks.append(true_risk)
            if raw:
              pred_risk = raw_risk_predictions[sample_id].get('Risk', '')
              # pred_risk = raw_risk_predictions[sample_id]['Risk']
              random_number = random.random()
              if random_number < 0.5:
                  pred_risk = 'Yes'
              else:
                  pred_risk = 'No'
              if 'yes' in pred_risk.lower():
                  pred_risk = 'Yes'
              elif 'no' in pred_risk.lower():
                  pred_risk = 'No'
              else:
                  logger.warning(f"Unexpected risk prediction for sample {sample_id}: {pred_risk}")
                  pred_risk = 'Yes' if 'risk' in pred_risk.lower() else 'No'
              pred_risks.append(pred_risk)
              risk_predictions[sample_id] = pred_risk
            else:
              # Get predicted intents for this sample
              if sample_id not in intent_predictions:
                  logger.warning(f"No intent predictions for sample {sample_id}")
                  continue

              intent_data = intent_predictions[sample_id]

              # Get scene graph if available
              scene_graph_data = {}
              if scene_graph_dir:
                  scene_graph_path = os.path.join(scene_graph_dir, f"scene_graph_{sample_id}.json")
                  if os.path.exists(scene_graph_path):
                      with open(scene_graph_path, 'r') as f:
                          scene_graph_data = json.load(f)

              # Create prompt for risk assessment
              prompt = f'For a given scene from the perspective of an autonomous vehicle, '

              if scene_graph_data:
                  prompt += f'this is the scene graph: {scene_graph_data}. '
              if gt:
                prompt += f'The ground truth intent of the objects in the scene is: {gt_sample["Pedestrians"], gt_sample["Cyclists"]}. '
              else:
                prompt += f'These are the intents of the objects in the scene: {intent_data}. '
              prompt += f'Given this information determine if this is a risky scene. Answer only with "Yes" or "No".A risky scene is one that warrants caution or attention from the driver. It does not necessarily mean a accident prone scene. "risk" is defined as the presence of objects or agents in a driving scene that may influence the future behavior of the ego-vehicle, particularly those that trigger a behavioral response such as braking.'

              # Generate risk prediction
              pred_risk = generate_response(prompt, api_key)

              if pred_risk is None:
                  logger.warning(f"Skipping sample {sample_id} due to response generation error")
                  continue

              # Clean up prediction to ensure it's Yes or No
              if 'yes' in pred_risk.lower():
                  pred_risk = 'Yes'
              elif 'no' in pred_risk.lower():
                  pred_risk = 'No'
              else:
                  logger.warning(f"Unexpected risk prediction for sample {sample_id}: {pred_risk}")
                  pred_risk = 'Yes' if 'risk' in pred_risk.lower() else 'No'

              pred_risks.append(pred_risk)
              risk_predictions[sample_id] = pred_risk  # Store the prediction

            logger.info(f"Sample {sample_id} - Predicted: {pred_risk}, Actual: {true_risk}")
            processed_samples += 1

        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            continue

    # Calculate metrics
    if processed_samples > 0:
        # Convert Yes/No to 1/0 for metric calculation
        y_true = [1 if r == 'Yes' else 0 for r in true_risks]
        y_pred = [1 if r == 'Yes' else 0 for r in pred_risks]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()

        logger.info(f"Processed {processed_samples} samples")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'processed_samples': processed_samples,
            'total_samples': num_samples,
            'risk_predictions': risk_predictions  # Include risk predictions in the results
        }
    else:
        logger.warning("No samples were processed")
        return {
            'accuracy': 0,
            'balanced_accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'processed_samples': 0,
            'total_samples': num_samples,
            'risk_predictions': {}  # Empty risk predictions
        }

def main():
    # File paths
    ground_truth_path = '<dataset-path>'
    intent_predictions_path = '<path-to-all_intent_jsons.json>'
    scene_graph_dir = '<path-to-all_scene_graphs.json>'  # Optional
    raw = False
    gt = True
    # Load ground truth data
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # Load intent predictions

    if raw:
      with open('<raw-pred-path>', 'r') as f:
        risk_predictions = json.load(f)
        intent_predictions = None
    else:
      with open(intent_predictions_path, 'r') as f:
        intent_predictions = json.load(f)


    # Evaluate risk
    risk_results = evaluate_risk(
        ground_truth,
        intent_predictions,
        scene_graph_dir=scene_graph_dir if os.path.exists(scene_graph_dir) else None,
        num_samples=len(intent_predictions),
        raw = raw,
        raw_risk_predictions = risk_predictions if raw else None,
        gt = gt
    )

    # Print results
    print("\n===== Risk Evaluation =====")
    print(f"Accuracy: {risk_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {risk_results['balanced_accuracy']:.4f}")
    print(f"Precision: {risk_results['precision']:.4f}")
    print(f"Recall: {risk_results['recall']:.4f}")
    print(f"F1 Score: {risk_results['f1_score']:.4f}")
    print(f"Confusion Matrix: {risk_results['confusion_matrix']}")
    print(f"Processed Samples: {risk_results['processed_samples']}/{risk_results['total_samples']}")

    # Save results to a file
    with open('<risk-results-json-path>', 'w') as f:
        json.dump(risk_results, f, indent=4)

    print("\n<risk-results-json-path>")

    # Save just the risk predictions to a separate file
    with open('<risk-predictions-json-path>', 'w') as f:
        json.dump(risk_results['risk_predictions'], f, indent=4)

    print("Risk predictions saved to <risk-predictions-json-path>")

if __name__ == "__main__":
    main()