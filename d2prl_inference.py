
import os
import cv2
import numpy as np
import pandas as pd
import torch

from torchvision import transforms
from tqdm import tqdm

import d2prl_src.d2prl as d2prl
import utils

def post_process_binary(mask, min_size=500):
    """
    Convert a predicted binary mask to cleaned version by thresholding and
    removing small connected components.
    """
    binary = mask > 0.5
    return utils.remove_small_components(binary, min_size=min_size)

def post_process_multiclass(mask_t, mask_s, min_size=500):
    """
    Refine target/source mask predictions and compute background mask.
    Applies component filtering and a smoothing filter to distinguish overlapping areas.
    """
    combined = (mask_t + mask_s) > 0
    cleaned = utils.remove_small_components(combined, min_size=min_size)

    mask_t = mask_t * cleaned
    mask_s = cleaned - mask_t
    pre_filter = mask_t - mask_s

    kernel = np.ones((50, 50), dtype=np.float32)
    filtered = cv2.filter2D(pre_filter.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)

    mask_t = (filtered > 0) * cleaned
    mask_s = cleaned.astype(np.float32) - mask_t.astype(np.float32)
    mask_b = 1.0 - (mask_t + mask_s)

    return mask_t, mask_s, mask_b


def predict_d2prl(model, image, input_size, device):
    """
    Run inference on a single image using the D2PRL model and return:
    - A binary mask (1 channel)
    - An RGB mask (3 channels: target, source, background)
    """
    # Define preprocessing steps: resize to model input size
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    # Store original resolution for resizing outputs later
    original_h, original_w = image.shape[:2]

    # Prepare image for model input
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Model inference (no gradient tracking)
    with torch.no_grad():
        binary_pred, target_pred, source_pred = model(image_tensor)

    # Helper to convert model output tensors to NumPy arrays
    def to_numpy(tensor):
        return tensor.permute(0, 2, 3, 1).squeeze().cpu().numpy()

    # Post-process 1-channel mask
    binary_np = to_numpy(binary_pred).round()
    binary_mask = post_process_binary(binary_np)
    binary_mask_resized = cv2.resize((binary_mask * 255).astype(np.uint8), (original_w, original_h))

    # Post-process 3-channel masks
    target_np = to_numpy(target_pred)
    source_np = to_numpy(source_pred)
    target_np[target_np > 0] = 1
    source_np[source_np > 0] = 1

    mask_t, mask_s, mask_b = post_process_multiclass(target_np, source_np)

    # Stack into RGB mask
    rgb_mask = np.stack([
        (mask_t * 255).astype(np.uint8),
        (mask_s * 255).astype(np.uint8),
        (mask_b * 255).astype(np.uint8)
    ], axis=2)
    rgb_mask_resized = cv2.resize(rgb_mask, (original_w, original_h))

    return binary_mask_resized, rgb_mask_resized


def load_model(params, device):
    """
    Load the D2PRL model and its weights from disk, checking compatibility.
    """
    model = d2prl.DPM(params['image_size'], params['batch_size'], params['pm_iterations'], is_trainning=False)

    if not os.path.isfile(params['model_path']):
        raise FileNotFoundError(f"Model weights not found at {params['model_path']}")

    # Load checkpoint and filter valid keys
    state = torch.load(params['model_path'], map_location=device)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in state.items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    model = model.to(device).eval()

    return model


def run_inference(params):
    """
    Run inference on an entire dataset based on a metadata CSV file.
    Generates and optionally saves both 1-channel and 3-channel mask predictions.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])
    np.random.seed(params['seed'])

    # Detect and prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load D2PRL model with pretrained weights
    model = load_model(params, device)
    print(f"Loaded model from {params['model_path']}")

    # Load dataset metadata
    metadata_path = os.path.join(params['dataset_path'], 'metadata.csv')
    df = pd.read_csv(metadata_path)
    df = df.sort_values(by=df.columns[0])

    # Loop over each image in the dataset
    for _, row in tqdm(df.iterrows(), total=len(df), desc="D2PRL inference"):
        input_path = os.path.join(params['dataset_path'], row['file_path'])
        mask_path  = os.path.join(params['dataset_path'], row['mask_path'])

        # Load image and ground truth mask
        input_image = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        gt_mask     = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

        # Perform model inference
        binary_mask, rgb_mask = predict_d2prl(model, input_image, params['image_size'], device)

        # Create output folder for results
        case_name = os.path.splitext(os.path.basename(row['file_path']))[0]
        case_dir  = os.path.join(params['output_path'], case_name)

        if params.get("save", False):
            os.makedirs(case_dir, exist_ok=True)

        # Plot and optionally save the result
        utils.plot_result(
            input_image, gt_mask, binary_mask,
            model_name="Original D2PRL",
            show=params.get("show", False),
            save=params.get("save", False),
            save_path=os.path.join(case_dir, case_name)
        )

if __name__ == "__main__":
    # Load config from JSON (inference mode)
    params = utils.load_params_from_json("params_configs.json", mode="inference")
    run_inference(params)

