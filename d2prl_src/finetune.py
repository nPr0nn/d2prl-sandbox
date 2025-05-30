import os
import time
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import pandas as pd 
from getitem2 import Config, Data
import models_D2PRL_train as models
from sklearn.model_selection import train_test_split


def dice_loss(y_true, y_pred, epsilon=1e-6):
    y_true = y_true.flatten(start_dim=1)
    y_pred = y_pred.flatten(start_dim=1)
    intersection = torch.sum(y_true * y_pred, dim=1)
    union = torch.sum(y_true, dim=1) + torch.sum(y_pred, dim=1)
    dice = 1 - (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DPM segmentation model")
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to pretrained model .pth')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--input_size', type=int, default=448, help='Random seed')
    parser.add_argument('--seed', type=int, default=42, help='Input image size')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Fraction of the data to validation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--pm_iter', type=int, default=32, help='PatchMatch iterations')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze SCSEUnet backbone')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N iterations')
    return parser.parse_args()

def main():
    args = parse_args()
    args.freeze_backbone = True

    # Prepare output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or f"finetune_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # read and split the CSV
    dataset_folder = os.path.dirname(args.dataset)
    df = pd.read_csv(args.dataset)
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=42, shuffle=True
    )
    train_csv = os.path.join(out_dir, "train_split.csv")
    val_csv   = os.path.join(out_dir, "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    cfg = Config(mode='train', batch=args.batch_size, size=args.input_size)
    train_dataset = Data(cfg, csv_path=train_csv, folder_path=dataset_folder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    cfg_val = Config(mode='valid', batch=args.batch_size, size=args.input_size)
    val_dataset = Data(cfg_val, csv_path=val_csv, folder_path=dataset_folder)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


    # Model initialization
    model = models.DPM(args.input_size, args.batch_size, args.pm_iter)

    # Load pretrained weights
    if args.pretrain and os.path.isfile(args.pretrain):
        print(f"Loading pretrained weights from {args.pretrain}")
        state = torch.load(args.pretrain, map_location='cpu')
        model_dict = model.state_dict()
        # filter and update only matching keys
        pretrained_dict = {k: v for k, v in state.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        for name, param in model.unet.named_parameters():
            param.requires_grad = False
        print("Backbone frozen. Fine-tuning only mask heads.")

    model = nn.DataParallel(model).to(device)

    # Optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, gts, _ in pbar:
            imgs = imgs.to(device)
            gts = gts.to(device)

            # prepare targets
            simi_gts = (1 - gts[:, 2, :, :]).unsqueeze(1)
            simi_gts2 = gts[:, 0, :, :].unsqueeze(1)
            simi_gts3 = gts[:, 1, :, :].unsqueeze(1)

            optimizer.zero_grad()
            with autocast('cuda'):
                out1, _, out3, out4 = model(imgs)
                loss1 = dice_loss(out1, simi_gts)
                loss2 = dice_loss(out4, simi_gts2)
                # example error-based loss
                er, _ = torch.max(torch.cat([
                    0.05 - out3 * (simi_gts3 - simi_gts2),
                    torch.zeros_like(out3)
                ], dim=1), dim=1, keepdim=True)
                loss3 = torch.sum(out1 * er * 1e-4)
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            global_step += 1
            pbar.set_postfix({'loss': np.mean(epoch_losses)})

            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(out_dir, f"step_{global_step}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, gts, _ in val_loader:
                imgs = imgs.to(device)
                gts = gts.to(device)
                simi_val = (1 - gts[:, 2, :, :]).unsqueeze(1)
                out1, _, _, out4 = model(imgs)
                v_loss = dice_loss(out1, simi_val)
                val_losses.append(v_loss.item())
        mean_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch} Validation Loss: {mean_val_loss:.5f}")

        # Save best
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_path = os.path.join(out_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved at epoch {epoch}")

    print("Training complete.")


if __name__ == "__main__":
    main()

