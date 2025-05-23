#!/usr/bin/env python3
import os
import json
import logging
import numpy as np
from tqdm import tqdm
from itertools import islice
import openai
from bert_score import score
# Optional: if you still want Sentence-Transformer cosine similarity
from sentence_transformers import SentenceTransformer, util
openai.api_key = '<api-key>'


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('action_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load Sentence-Transformer model (optional)
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_response(prompt: str) -> str:
    """
    Generate a single response from OpenAI's chat endpoint.
    """
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes traffic scenes and suggests appropriate driving actions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None

def calculate_semantic_similarity(a: str, b: str) -> float:
    """
    (Optional) compute cosine similarity between two texts using Sentence-Transformers.
    """
    emb1 = st_model.encode(a, convert_to_tensor=True)
    emb2 = st_model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def evaluate_suggested_actions(
    ground_truth: dict,
    intent_predictions: dict,
    scene_graphs: dict = None,
    risk_predictions: dict = None,
    num_samples: int = 100,
    raw = False,
    raw_sa_predictions = None
) -> dict:
    """
    For each sample, generate an action via the LLM, then compute BERTScore
    between the ground-truth 'suggested_action' and the generated text.
    """
    true_actions = []
    pred_actions = []
    semantic_sims = []

    for sample_id, gt_sample in tqdm(islice(ground_truth.items(), num_samples), total=num_samples):
        true_act = gt_sample.get('suggested_action', '').strip()
        if not true_act or true_act.upper() == "N/A":
            logger.debug(f"Skipping {sample_id}: no ground-truth action")
            continue
        if not raw and sample_id not in intent_predictions:
            logger.debug(f"Skipping {sample_id}: no intent preds")
            continue

        # build prompt

        if raw:
          pred_act = raw_sa_predictions[sample_id]['Suggested_action']

        else:
          prompt = (
              "For a given autonomous-driving scene, "
              f"these are the object intents: {intent_predictions[sample_id]}. "
          )
          if scene_graphs and sample_id in scene_graphs:
              prompt += f"Scene graph: {scene_graphs[sample_id]}. "
          if risk_predictions and sample_id in risk_predictions:
              prompt += ("The scene is risky. " if risk_predictions[sample_id] == "Yes"
                        else "The scene is not risky. ")
          prompt += "What action should the vehicle take? Be concise and specific."
          pred_act = generate_response(prompt)
        if not pred_act:
            logger.warning(f"No response for {sample_id}")
            continue

        true_actions.append(true_act)
        pred_actions.append(pred_act)
        semantic_sims.append(calculate_semantic_similarity(true_act, pred_act))

        logger.info(f"[{sample_id}] GT: {true_act}  →  PRED: {pred_act}")

    # Compute BERTScore
    P, R, F1 = score(
        true_actions,
        pred_actions,
        lang="en",
        verbose=True
    )
    avg_p = P.mean().item()
    avg_r = R.mean().item()
    avg_f1 = F1.mean().item()
    print(len(F1))
    # Choose a threshold to determine "semantically similar"
    # This is a decision you make based on your requirements
    similarity_threshold = 0.8  # Example threshold

    # Create binary classifications based on the threshold
    correct_predictions = (F1 >= similarity_threshold).float()

    # Calculate accuracy (percentage of samples above threshold)
    accuracy = correct_predictions.mean().item()

    print(f"Accuracy at threshold {similarity_threshold}: {accuracy:.4f}")
    # Distribution of F1 above thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    f1_dist = {
        f"above_{int(t*100)}": (F1 >= t).float().mean().item()
        for t in thresholds
    }

    return {
        "processed_samples": len(true_actions),
        "average_precision": avg_p,
        "average_recall":    avg_r,
        "average_f1":        avg_f1,
        "f1_threshold_distribution": f1_dist,
        # Optional extras:
        "semantic_similarity_mean": np.mean(semantic_sims) if semantic_sims else None,
        "true_actions":  true_actions,
        "pred_actions":  pred_actions,
    }

def main():
    # Paths (adjust as needed)
    gt_path     = '<dataset-path>'
    intent_path = '<path-to-all_intent_jsons.json>'
    sg_path     = '<path-to-all_scene_graphs.json>'
    risk_path   = '<path-to-risk_predictions.json>'
    raw = True
    with open(gt_path)     as f: ground_truth      = json.load(f)
    if raw:
      with open('<path-to-raw-op>', 'r') as f:
        sugg_acc_raw = json.load(f)
        intent_predictions = None
        scene_graphs = None
    else:

      with open(intent_path) as f: intent_predictions = json.load(f)
      scene_graphs = json.load(open(sg_path)) if os.path.exists(sg_path) else None
    risk_preds   = json.load(open(risk_path)) if os.path.exists(risk_path) else None

    results = evaluate_suggested_actions(
        ground_truth,
        intent_predictions,
        scene_graphs=scene_graphs,
        risk_predictions=risk_preds,
        num_samples=len(risk_preds),
        raw = raw,
        raw_sa_predictions = sugg_acc_raw if raw else None
    )

    # Print out BERTScore metrics
    print("\n===== BERTScore-based Action Evaluation =====")
    print(f"Processed Samples      : {results['processed_samples']}")
    print(f"Avg Precision (P)      : {results['average_precision']:.4f}")
    print(f"Avg Recall    (R)      : {results['average_recall']:.4f}")
    print(f"Avg F1        (F1)     : {results['average_f1']:.4f}")
    print("F1 Threshold Distribution:")
    for k,v in results["f1_threshold_distribution"].items():
        print(f"  {k}: {v:.2%}")
    if results.get("semantic_similarity_mean") is not None:
        print(f"\n(Optional) Avg Cosine-Similarity: {results['semantic_similarity_mean']:.4f}")

    # Save full results
    output_path = '<path-to-results-json>'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
