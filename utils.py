
import os
import cv2
import numpy as np 
import json

from matplotlib import pyplot as plt 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ==== Input Parsing ==== 
def load_params_from_json(config_path, mode='inference'):
    """
    Load parameters from a JSON config file with support for different modes.
    Args:
        config_path (str): Path to the JSON file.
        mode (str): Mode to load ('inference' or 'finetune').
    Returns:
        dict: Dictionary of parameters for the specified mode.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    if mode not in config_data:
        raise KeyError(f"Mode '{mode}' not found in config file. Available modes: {list(config_data.keys())}")

    return config_data[mode]

# ==== Pre-processing ==== 
def remove_small_components(mask, min_size=500):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 1

    return output.astype(bool)

# ==== Results Related ==== 
def plot_result(input, ground_truth, prediction, model_name, show=True, save=False, save_path=None, figsize=(12,4)):
    plt.figure( figsize=figsize )
    plt.subplot(131)
    plt.imshow(input)
    plt.title('Original Image')
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(ground_truth)
    plt.subplot(133)
    plt.imshow(prediction) 
    plt.title(model_name + ' Prediction')
    if(save): plt.savefig(save_path) 
    if(show): plt.show()

def compute_classification_metrics(ground_truth, prediction):
    # Flatten the masks
    ground_truth = np.asarray(ground_truth, dtype=bool).flatten()
    prediction   = np.asarray(prediction, dtype=bool).flatten()

    # Compute metrics
    accuracy  = accuracy_score(ground_truth, prediction) 
    precision = precision_score(ground_truth, prediction, zero_division=0)
    recall    = recall_score(ground_truth, prediction)
    f1        = f1_score(ground_truth, prediction)

    return {
        "Accuracy": accuracy, 
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
