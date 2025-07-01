
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from torch.utils.checkpoint import checkpoint

# Add current script directory to sys.path
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

# Local imports
from deep_pm import PatchMatch
from scse import SCSEUnet

def weights_init_normal(m):
    """
    Initializes the weights of a given PyTorch module using Kaiming Normal initialization 
    for Conv2d and Linear layers, and Normal initialization for BatchNorm2d layers.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0.0)

class DLFError(nn.Module):
    """
    Dense Linear Fitting Error module that computes transformation
    using predefined convolutional kernels loaded from a MATLAB `.mat` file.

    This module splits the input tensor into two halves along the channel dimension,
    applies four shared convolutional filters to both halves, and computes an error term.

    Args:
        vv_mvf_path (str): Path to the .mat file containing 'VV_mvf' convolution kernels.
        requires_grad (bool): Whether the convolution kernels and biases should require gradients.
    """
    def __init__(self, vv_mvf_path, requires_grad):
        super().__init__()

        vv_mvf = sio.loadmat(vv_mvf_path)['VV_mvf']
        vv_mvf = torch.tensor(vv_mvf, dtype=torch.float32)  # shape: (k_size, _, _, 4)

        k_size = vv_mvf.shape[0]
        output_size = 1
        vv_mvf = vv_mvf.repeat(output_size, 1, 1, 1, 1)

        # Extract filters for each output
        self.VV_1 = nn.Parameter(vv_mvf[..., 0], requires_grad=requires_grad)
        self.VV_2 = nn.Parameter(vv_mvf[..., 1], requires_grad=requires_grad)
        self.VV_3 = nn.Parameter(vv_mvf[..., 2], requires_grad=requires_grad)
        self.VV_4 = nn.Parameter(vv_mvf[..., 3], requires_grad=requires_grad)

        # Biases
        self.bias_1 = nn.Parameter(torch.zeros(output_size), requires_grad=requires_grad)
        self.bias_2 = nn.Parameter(torch.zeros(output_size), requires_grad=requires_grad)
        self.bias_3 = nn.Parameter(torch.zeros(output_size), requires_grad=requires_grad)
        self.bias_4 = nn.Parameter(torch.zeros(output_size), requires_grad=requires_grad)

        self.pad_size = k_size // 2  # Same as floor(k_size / 2)

    def _apply_convs(self, input):
        sq_input = input * input
        conv1 = F.conv2d(sq_input, self.VV_1, self.bias_1, padding=self.pad_size)
        conv2 = F.conv2d(input, self.VV_2, self.bias_2, padding=self.pad_size)
        conv3 = F.conv2d(input, self.VV_3, self.bias_3, padding=self.pad_size)
        conv4 = F.conv2d(input, self.VV_4, self.bias_4, padding=self.pad_size)
        return conv1 - (conv2**2 + conv3**2 + conv4**2)

    def forward(self, input):
        """
        Splits the input into two parts along the channel dimension, applies 
        convolutions, and returns their combined error.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C, H, W), where C is even.

        Returns:
            torch.Tensor: Output tensor containing the combined error.
        """
        batch_size, channels, height, width = input.shape
        assert channels % 2 == 0, "Input channels must be even."
        
        error1 = self._apply_convs(input[:, :channels // 2, :, :])
        error2 = self._apply_convs(input[:, channels // 2:, :, :])

        return error1 + error2

class ZMPolarConv(nn.Module):
    """
    Zernike Moment Polar Convolution Layer.

    This layer performs a complex-valued convolution using precomputed Zernike moment
    polar filters (real and imaginary parts), loaded from a MATLAB `.mat` file.
    The real and imaginary parts are convolved separately, and their magnitudes
    are combined via sqrt(real² + imag² + ε).

    Args:
        zm_polar_path (str): Path to the .mat file containing 'ZM_polar_k' kernels.
                             The array is expected to be of shape (k_size, k_size, 2 * output_channels).
        input_size (int): Number of input channels (used to repeat filters).
        requires_grad (bool): Whether the filter weights and biases require gradients.
    """
    def __init__(self, zm_polar_path, input_size, requires_grad):
        super().__init__()

        zm_data = sio.loadmat(zm_polar_path)['ZM_polar_k']  # Shape: (k, k, 2*out_channels)
        k_size, _, total_channels = zm_data.shape
        output_size = total_channels // 2

        zm_real = zm_data[:, :, :output_size]
        zm_imag = zm_data[:, :, output_size:]

        real_tensor = torch.tensor(zm_real, dtype=torch.float32)
        imag_tensor = torch.tensor(zm_imag, dtype=torch.float32)

        # Repeat and reshape to match (out_channels, in_channels, kH, kW)
        real_tensor = real_tensor.repeat(input_size, 1, 1, 1).permute(3, 0, 1, 2)
        imag_tensor = imag_tensor.repeat(input_size, 1, 1, 1).permute(3, 0, 1, 2)

        self.real_weight = nn.Parameter(real_tensor, requires_grad=requires_grad)
        self.imag_weight = nn.Parameter(imag_tensor, requires_grad=requires_grad)

        self.real_bias = nn.Parameter(torch.zeros(output_size), requires_grad=requires_grad)
        self.imag_bias = nn.Parameter(torch.zeros(output_size), requires_grad=requires_grad)

        self.pad_size = k_size // 2  # Same as floor(k_size / 2)

    def forward(self, input_tensor):
        """
        Apply compleinput_tensor convolution using real and imaginary parts separately,
        then combine using magnitude.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, output_channels, H, W)
        """
        input_tensor_real = F.conv2d(input_tensor, self.real_weight, self.real_bias, padding=self.pad_size)
        input_tensor_imag = F.conv2d(input_tensor, self.imag_weight, self.imag_bias, padding=self.pad_size)

        # Combine real and imaginary parts
        magnitude = torch.sqrt(input_tensor_real**2 + input_tensor_imag**2 + 1e-10)
        return magnitude


class DPM(nn.Module):
    """
    Deep PatchMatch Network
    """
    def __init__(self, size, batch_size, pmiter, is_trainning=False):
        super().__init__()
        self.is_trainning = is_trainning

        # Architeture Modules
        self.patchmatch = PatchMatch(pmiter, 50, size, batch_size)
        self.ZM_conv    = ZMPolarConv(os.path.join(this_dir, "ZM_polar_k13.mat"), 3, requires_grad=False)

        self.head_mask = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
        )
        self.head_mask2 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.head_mask3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.head_mask4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )

        self.last_mask = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.last_mask2 = nn.Sequential(
            nn.Conv2d(1344, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        self.DLFerror11 = DLFError(os.path.join(this_dir, "VV_mvf11.mat"), requires_grad=False)
        self.DLFerror7  = DLFError(os.path.join(this_dir, "VV_mvf7.mat"), requires_grad=False)
        self.DLFerror9  = DLFError(os.path.join(this_dir, "VV_mvf9.mat"), requires_grad=False)
        
        self.unet       = SCSEUnet(backbone_arch='senet154', num_channels=3) 
       
        # Normal initialization of network weights
        for module in [
            self.head_mask, self.head_mask2,
            self.head_mask3, self.head_mask4,
            self.last_mask, self.last_mask2
        ]:
            module.apply(weights_init_normal)

    def forward(self, input_tensor):
        """
        Forward pass to compute a refined binary mask using multi-scale Zernike moments and CNN features.
        """
        _, _, height, width = input_tensor.shape

        # === 1. Generate Multi-Scale Inputs ===
        scale_down = F.interpolate(input_tensor, size=(int(height / 1.5), int(width / 1.5)), mode='bilinear', align_corners=True)
        scale_up   = F.interpolate(input_tensor, size=(int(height / 0.75), int(width / 0.75)), mode='bilinear', align_corners=True)

        # === 2. Compute Zernike Moments at All Scales ===
        zm_base  = self.ZM_conv(input_tensor)
        zm_small = F.interpolate(self.ZM_conv(scale_down), size=(height, width), mode='bilinear', align_corners=True)
        zm_large = F.interpolate(self.ZM_conv(scale_up), size=(height, width), mode='bilinear', align_corners=True)

        # === 3. Extract CNN Features (Padded) from All Scales ===
        apply_mask_with_padding = lambda x: self.head_mask(F.pad(x, pad=[7, 7, 7, 7], mode='reflect'))

        cnn_base  = apply_mask_with_padding(input_tensor)
        cnn_small = F.interpolate(apply_mask_with_padding(scale_down), size=(height, width), mode='bilinear', align_corners=True)
        cnn_large = F.interpolate(apply_mask_with_padding(scale_up), size=(height, width), mode='bilinear', align_corners=True)

        # === 4. Concatenate Multi-Scale Representations ===
        zm_features_concat  = torch.cat([zm_large, zm_base, zm_small], dim=-3)
        cnn_features_concat = torch.cat([cnn_large, cnn_base, cnn_small], dim=-3)
        cnn_features_detached = cnn_features_concat.clone().detach_()

        # === 5. PatchMatch to Get Correspondences and Offsets ===
        coords, offsets = self.patchmatch(zm_features_concat.half(), cnn_features_concat.half(), cnn_features_detached.half())
        x_zm, y_zm, x_cnn, y_cnn = coords
        dx_zm, dy_zm, dx_cnn, dy_cnn = offsets

        # === 6. Dense Linear Fitting Errors ===
        def dlf_activation(x, y, dlf):
            return 2 * torch.sigmoid(1 / (dlf(torch.cat([x, y], dim=-3)) + 1e-10)) - 1.0

        zm_dlf_errors  = [dlf_activation(x_zm, y_zm, layer) for layer in [self.DLFerror7, self.DLFerror9, self.DLFerror11]]
        cnn_dlf_errors = [dlf_activation(x_cnn, y_cnn, layer) for layer in [self.DLFerror7, self.DLFerror9, self.DLFerror11]]

        # === 7. Combine All Features ===
        combined_offsets = torch.cat([dx_zm, dy_zm, *zm_dlf_errors, dx_cnn, dy_cnn, *cnn_dlf_errors], dim=-3)

        # === 8. Predict Last Mask and UNet Mask ===
        last_mask = self.last_mask(combined_offsets)
        unet_mask    = self.unet(input_tensor)

        # === 9. Merge UNet and Last Mask at Inference ===
        if not self.is_trainning:
            has_overlap = lambda a, b: torch.sum(a * b) > 0.5 * torch.sum(b)
            merge_masks = lambda a, b: torch.where(a > b, a, b) if has_overlap(a, b) else a
            last_mask = merge_masks(last_mask, unet_mask)

        # === 10. Detach and Binarize Mask ===
        binarized_mask = last_mask.clone().detach_().round()

        # === 11. Detach Coordinates (for safety) ===
        for coord in [x_zm, y_zm, x_cnn, y_cnn]:
            coord.detach_()

        # === 12. Extract Hierarchical CNN Features ===
        head1 = self.head_mask2(input_tensor)
        head2 = self.head_mask3(head1)
        head3 = self.head_mask4(head2)

        # === 13. Normalize Coordinates (from pixel indices to [-1, 1]) ===
        def normalize_coords(x, y):
            norm = lambda t, dim: (t - (t.size(dim) - 1) / 2) / ((t.size(dim) - 1) / 2)
            return norm(x, 3), norm(y, 2)

        x_zm_norm, y_zm_norm     = normalize_coords(x_zm, y_zm)
        x_cnn_norm, y_cnn_norm   = normalize_coords(x_cnn, y_cnn)

        # === 14. Build Sampling Grids ===
        def build_grid(x, y):
            grid = torch.cat((x.unsqueeze(4), y.unsqueeze(4)), dim=4)
            return grid.view(grid.size(0) * grid.size(1), *grid.shape[2:])

        grid_zm   = build_grid(x_zm_norm, y_zm_norm)
        grid_cnn  = build_grid(x_cnn_norm, y_cnn_norm)

        # === 15. Fuse Coordinates Using Mask Weights ===
        sampled_mask = F.grid_sample(binarized_mask, grid_zm, align_corners=True)
        mask_weight  = binarized_mask * sampled_mask
        inverse_mask = 1 - mask_weight

        x_fused = x_cnn_norm * mask_weight.round() + x_zm_norm * inverse_mask.round()
        y_fused = y_cnn_norm * mask_weight.round() + y_zm_norm * inverse_mask.round()

        # === 16. Upsample Everything to Match Final Feature Resolution ===
        _, _, head3_height, head3_width = head3.shape
        resize_to_head3_size = lambda x: F.interpolate(x, size=(head3_height, head3_width), mode='bilinear', align_corners=True)

        binarized_mask_up  = resize_to_head3_size(binarized_mask).round()
        x_zm_up, y_zm_up   = resize_to_head3_size(x_zm_norm), resize_to_head3_size(y_zm_norm)
        x_cnn_up, y_cnn_up = resize_to_head3_size(x_cnn_norm), resize_to_head3_size(y_cnn_norm)
        x_fused_up         = resize_to_head3_size(x_fused)
        y_fused_up         = resize_to_head3_size(y_fused)

        head1_up = resize_to_head3_size(head1)
        head2_up = resize_to_head3_size(head2)
        head_features = torch.cat([head1_up, head2_up, head3], dim=-3)

        # === 17. Sample Features from All Grids ===
        grid_zm_up    = build_grid(x_zm_up, y_zm_up)
        grid_cnn_up   = build_grid(x_cnn_up, y_cnn_up)
        grid_fused_up = build_grid(x_fused_up, y_fused_up)

        feat_zm  = F.grid_sample(head_features, grid_zm_up, align_corners=True)
        feat_cnn = F.grid_sample(head_features, grid_cnn_up, align_corners=True)

        # === 18. Fuse Feature Maps Using Sampled Inputs and Mask ===
        fused_features = torch.cat([
            head_features * binarized_mask_up,
            feat_zm * binarized_mask_up,
            feat_cnn * binarized_mask_up
        ], dim=-3)

        # === 19. Final Mask Refinement ===
        refined_mask = self.last_mask2(fused_features)
        refined_mask = refined_mask * binarized_mask_up + (1 - binarized_mask_up) * 0.5

        # === 20. Compute Residuals Between Refined and Sampled ===
        sampled_refined    = F.grid_sample(refined_mask, grid_fused_up, align_corners=True)
        refined_mask_up    = F.interpolate(refined_mask, size=(height, width), mode='bilinear', align_corners=True)
        sampled_refined_up = F.interpolate(sampled_refined, size=(height, width), mode='bilinear', align_corners=True)

        residual     = refined_mask_up - sampled_refined_up
        residual_neg = -1 * (residual * binarized_mask).clamp_max(0)
        residual_pos = (residual * binarized_mask).clamp_min(0)

        # === 21. Final Output ===
        final_output = torch.cat([residual_neg, residual_pos, 1 - binarized_mask], dim=-3)

        if self.is_trainning:
            return last_mask, binarized_mask, residual, unet_mask

        return last_mask, residual_neg, residual_pos

