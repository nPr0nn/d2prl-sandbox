
import os
import cv2
import numpy as np 

from matplotlib import pyplot as plt 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def plot_result(input, ground_truth, pred, model_name, show=True, save=False, save_path=None, figsize=(12,4)):
    """
    Visualize input, ground truth, and predicted result
    """
    plt.figure( figsize=figsize )
    plt.subplot(131)
    plt.imshow(input)
    plt.title('Original Image')
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(ground_truth)
    plt.subplot(133)
    plt.imshow(pred) 
    plt.title(model_name + ' Prediction')
    if(save): plt.savefig(save_path) 
    if(show): plt.show()

def compute_classification_metrics(ground_truth, prediction):
    """
    Computes common evaluation metrics for binary masks.

    Parameters:
        ground_truth (np.ndarray): Binary mask for the ground truth (0 or 1).
        prediction (np.ndarray): Binary mask for the prediction (0 or 1).

    Returns:
        dict: A dictionary containing Accuracy, Precision, Recall, F1
    """

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

def remove_small_components(mask, min_size=500):
    """
    Removes connected components smaller than `min_size` from a binary mask.
    
    Args:
        mask (ndarray): Binary mask (bool or 0/1 values).
        min_size (int): Minimum size (in pixels) for components to keep.

    Returns:
        ndarray: Binary mask with small components removed.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 1

    return output.astype(bool)
