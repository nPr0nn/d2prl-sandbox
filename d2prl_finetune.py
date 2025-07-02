
import os
import cv2
import numpy as np
import pandas as pd
import torch
import time
import json
import csv 

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import d2prl_src.getitem as getitem
import d2prl_src.d2prl as d2prl
import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    Compute Dice loss between ground truth and prediction.
    """
    y_true       = y_true.flatten(start_dim=1)
    y_pred       = y_pred.flatten(start_dim=1)
    intersection = torch.sum(y_true * y_pred, dim=1)
    union        = torch.sum(y_true, dim=1) + torch.sum(y_pred, dim=1)
    dice         = 1 - (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean()


def load_model(params, device):
    """
    Load the D2PRL model and optionally freeze its backbone.
    """
    model = d2prl.DPM(params['image_size'], params['batch_size'], params['pm_iterations'], is_trainning=True)

    if not os.path.isfile(params['model_path']):
        raise FileNotFoundError(f"Model weights not found at {params['model_path']}")

    checkpoint = torch.load(params['model_path'], map_location=device)
    model_dict = model.state_dict()
    compatible_weights = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(compatible_weights)
    model.load_state_dict(model_dict)

    if params.get('freeze_backbone', False):
        for name, param in model.unet.named_parameters():
            param.requires_grad = False
        print("Backbone frozen. Only training mask heads.")
    
    model = torch.nn.DataParallel(model).to(device)
    
    return model


def run_finetune(params):
    """
    Fine-tunes the D2PRL model on a custom dataset with supervision masks.
    """
    # Set seeds for reproducibility
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])
    np.random.seed(params['seed'])

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable cuDNN autotuner for potentially better performance
    torch.backends.cudnn.benchmark = True

    # Load and prepare model
    model = load_model(params, device)
    print(f"Loaded model from {params['model_path']}")

    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(params['output_path'], f"experiment_{timestamp}")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save hyperparameters to file
    with open(os.path.join(output_dir, "hyperparameters_used.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Setup CSV log file for losses
    loss_log_path = os.path.join(output_dir, "losses.csv")
    with open(loss_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    # Read and split CSV into train/val sets
    metadata_path = os.path.join(params['dataset_path'], 'metadata.csv')
    df = pd.read_csv(metadata_path)
    train_df, val_df = train_test_split(df, test_size=params['val_fraction'], random_state=params['seed'], shuffle=True)

    train_csv = os.path.join(output_dir, "train.csv")
    val_csv   = os.path.join(output_dir, "val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)

    # Configure datasets and loaders
    train_cfg = getitem.Config(mode='train', batch=params['batch_size'], size=params['image_size'])
    val_cfg   = getitem.Config(mode='valid', batch=params['batch_size'], size=params['image_size'])

    train_set = getitem.Data(train_cfg, csv_path=train_csv, folder_path=params['dataset_path'])
    val_set   = getitem.Data(val_cfg,   csv_path=val_csv,   folder_path=params['dataset_path'])

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_set,   batch_size=params['batch_size'], shuffle=False, num_workers=4)

    # Set up optimizer, scheduler, and scaler
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=params['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler    = GradScaler()

    best_val_loss    = float('inf')
    patience_counter = 0

    # === Training loop ===
    train_loss = []
    val_loss   = []
    for epoch in range(1, params['epochs'] + 1):
        model.train()
        epoch_train_losses = []
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{params['epochs']}")
        for image, gt_mask, _ in pbar:
            image, gt_mask = image.to(device), gt_mask.to(device)

            # Break ground truth mask into its components
            mask_tampering = (1 - gt_mask[:, 2, :, :]).unsqueeze(1)  # not blue
            mask_target    = gt_mask[:, 0, :, :].unsqueeze(1)        # red
            mask_source    = gt_mask[:, 1, :, :].unsqueeze(1)        # green

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                binary_mask, binary_mask_round, residual, unet_mask = model(image)

                loss1 = dice_loss(binary_mask, mask_tampering)
                loss2 = dice_loss(unet_mask, mask_target)

                error_term, _ = torch.max(torch.cat([
                    0.05 * torch.ones_like(residual) - residual * (mask_source - mask_target),
                    torch.zeros_like(residual)
                ], dim=-3), dim=1, keepdim=True)

                loss3 = torch.sum(binary_mask_round * mask_tampering * error_term * 1e-4)
                total_loss = loss1 + loss2 + loss3

            if np.isnan(total_loss.item()):
                continue

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()

            epoch_train_losses.append(total_loss.item())
            pbar.set_postfix({'Train Loss': np.mean(epoch_train_losses)})

        scheduler.step()

        # Save model after a specified number of epochs
        if epoch % params['saving_interval'] == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # === Validation ===
        model.eval()
        epoch_val_losses = []
        vbar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch}/{params['epochs']}")
        with torch.no_grad():
            for image, gt_mask, _ in vbar:
                image, gt_mask = image.to(device), gt_mask.to(device)
                mask_tampering = (1 - gt_mask[:, 2, :, :]).unsqueeze(1)

                binary_mask, _, _, _ = model(image)
                loss = dice_loss(binary_mask, mask_tampering)

                if np.isnan(loss.item()):
                    continue

                epoch_val_losses.append(loss.item())
                vbar.set_postfix({'Val Loss': np.mean(epoch_val_losses)})

        mean_train_loss = np.mean(epoch_train_losses)
        mean_val_loss   = np.mean(epoch_val_losses)
        train_loss.append(mean_train_loss)
        val_loss.append(mean_val_loss)

        print(f"Epoch {epoch} -- Validation Loss: {mean_val_loss:.5f}")

        # Log to CSV
        with open(loss_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, mean_train_loss, mean_val_loss])

        # Save best model
        if mean_val_loss < best_val_loss:
            patience_counter = 0
            best_val_loss = mean_val_loss
            best_path = os.path.join(output_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved at epoch {epoch}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{params['patience']}")
            if patience_counter >= params['patience']:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    print("Training complete :D")


    # === Plot and save losses ===
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    params = utils.load_params_from_json("params_configs.json", mode="finetune")
    run_finetune(params)
