
import os
import cv2
import sys
import argparse
import pandas
import numpy as np
import torch 

from torchvision import transforms
from tqdm import tqdm

import d2prl_src.d2prl as d2prl
import utils

def get_static_params():
    model_path    = "d2prl_weights.pth"
    dataset_path  = "/home/lucas/Pesquisa/Horus-IC/datasets/copy-move/recod-cmfd-fine-tunning-dataset/base"
    output_path   = "output"
    image_size    = 448
    batch_size    = 1
    pm_iterations = 40
    seed          = 42
    show          = True 
    save          = True

    return {
         "dataset_path": dataset_path,
         "model_path": model_path,
         "output_path": output_path,
         "image_size": image_size,
         "batch_size": batch_size,
         "pm_iterations": pm_iterations,
         "seed": seed,
         "show": show,
         "save": save,
     }        

def get_cli_params():
    parser = argparse.ArgumentParser(description="CLI args for D2PRL inference")
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (e.g folder/model.pth)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to testing dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--imgsz', type=int, default=448, help='Image size (default: 448)')
    parser.add_argument('--batchsz', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--pmiter', type=int, default=40, help='Patch Match number of iterations (default: 40)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--show', action='store_true', help='Should show the results')
    parser.add_argument('--save', action='store_true', help='Should save the results')
    args = parser.parse_args()

    # Paths validations
    if not os.path.isfile(args.model):
        print(f"Error: The specified file path for {args.model} does not exist")
        sys.exit(1)
    if not os.path.isdir(args.dataset):
        print(f"Error: The specified directory for {args.dataset} does not exist")
        sys.exit(1)
    if not os.path.isdir(args.output):
        print(f"Error: The specified directory for {args.output} does not exist")
        sys.exit(1)

    return {
         "dataset_path": args.dataset,
         "model_path": args.model,
         "output_path": args.output_path,
         "image_size": args.imgsz,
         "batch_size": args.batchsz,
         "pm_iterations": args.pmiter,
         "seed": args.seed,
         "show": args.show,
         "save": args.save,
     }        

def post_1c(mask):
    binary_mask = mask > 0.5
    return utils.remove_small_components(binary_mask, min_size=500)

def post_3c(mask_t, mask_s):
    combined_mask = (mask_t + mask_s) > 0
    cleaned_mask = utils.remove_small_components(combined_mask, min_size=500)
    mask_t = mask_t * cleaned_mask
    mask_s = cleaned_mask - mask_t
    pre_filter_mask = mask_t - mask_s
    kernel = np.ones((50, 50), dtype=np.float32)
    filtered = cv2.filter2D(pre_filter_mask.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    mask_t = (filtered > 0) * cleaned_mask
    mask_s = cleaned_mask.astype(np.float32) - mask_t.astype(np.float32)
    mask_b = 1.0 - (mask_t + mask_s)
    return mask_t, mask_s, mask_b

def d2prl_predict(model, input_image, imgsz, device='cuda'):
    """
    Runs D2PRL model inference on an input image and returns both binary and RGB masks.
    """
    # Pre-processing pipeline: convert to PIL, resize, and convert back to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])

    # Stores image original size
    h_orig, w_orig, _ = input_image.shape

    # Preprocess and move to device
    image_tensor = transform(input_image).unsqueeze(0).to(device)

    # D2PRL Inference
    with torch.no_grad():
        binary_pred, target_pred, source_pred = model(image_tensor)

    # Helper to convert prediction tensor to squeezed NumPy array
    to_numpy = lambda x: x.permute(0, 2, 3, 1).squeeze().cpu().numpy()

    # Process binary mask
    binary_np = to_numpy(binary_pred).round()
    binary_mask = post_1c(binary_np)
    binary_mask_resized = cv2.resize((binary_mask * 255).astype(np.uint8), (w_orig, h_orig))

    # Process target and source
    target_np = to_numpy(target_pred)
    source_np = to_numpy(source_pred)
    target_np[target_np > 0] = 1
    source_np[source_np > 0] = 1

    # Post-process and split into masks
    target_mask, source_mask, background_mask = post_3c(target_np, source_np)

    # Stack into RGB mask
    rgb_mask = np.stack([
        target_mask.astype(np.uint8) * 255,
        source_mask.astype(np.uint8) * 255,
        background_mask.astype(np.uint8) * 255
    ], axis=2)
    rgb_mask_resized = cv2.resize(rgb_mask, (w_orig, h_orig))

    return binary_mask_resized, rgb_mask_resized


def run_inference(params):
    """
    Runs D2PRL model inference on an dataset specified by a metadata.csv file contained in the dataset path
    """
    # Seed frameworks 
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])

    # Get torch device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device      = torch.device(device_name)

    # Load D2PRL model 
    d2prl_model = d2prl.DPM(params['image_size'], params['batch_size'], params['pm_iterations'])
    model_dict = d2prl_model.state_dict()
    for k, v in torch.load(params['model_path']).items():
        model_dict.update({k: v})
    d2prl_model.load_state_dict(model_dict)
    d2prl_model = d2prl_model.cuda()
    d2prl_model.eval()

    # Load dataset paths and run D2PRL on files
    dataframe = pandas.read_csv(os.path.join(params['dataset_path'], 'metadata.csv'))
    dataframe = dataframe.sort_values(by=dataframe.columns[0])
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc='Running D2PRL inference in dataset'):
        # Read images paths and read data
        input_image_path = os.path.join(params['dataset_path'], row['file_path'])
        gt_image_path    = os.path.join(params['dataset_path'], row['mask_path'])
        input_image      = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
        gt_image         = cv2.cvtColor(cv2.imread(gt_image_path), cv2.COLOR_BGR2GRAY)

        # Call model on input image
        pred1c_image, pred3c_image = d2prl_predict(d2prl_model, input_image, params['image_size'], device)

        # Plot results
        utils.plot_result(input_image, gt_image, pred1c_image, "Original D2PRL")

if __name__ == "__main__":
    params = get_static_params()
    run_inference(params)
