from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class RandomSampler(nn.Module):
    def __init__(self):
        super().__init__()
        # Multiplier: 0 and 1 reshaped for broadcasting
        self.range_multiplier = torch.arange(0.0, 2.0, 1).view(2, 1, 1)

    def forward(self, min_x, max_x, min_y, max_y, offset_x, offset_y):
        device = offset_x.device
        batch_size, _, h, w = min_x.shape
        range_multiplier = self.range_multiplier.to(device)

        noise = torch.rand((batch_size, 2, h, w), device=device)

        # Apply the noise in the range [min, max]
        delta_x = ((max_x - min_x) / 2) * (range_multiplier + noise)
        delta_y = ((max_y - min_y) / 2) * (range_multiplier + noise)

        sampled_x = min_x + delta_x
        sampled_y = min_y + delta_y

        sampled_x = sampled_x.unsqueeze(1).expand(-1, sampled_y.size(1), -1, -1, -1)
        sampled_y = sampled_y.unsqueeze(2).expand(-1, -1, sampled_y.size(1), -1, -1)

        sampled_x = sampled_x.contiguous().view(batch_size, -1, h, w)
        sampled_y = sampled_y.contiguous().view(batch_size, -1, h, w)

        # Concatenate with original offsets
        offset_x = torch.cat((sampled_x, offset_x), dim=1)
        offset_y = torch.cat((sampled_y, offset_y), dim=1)

        return offset_x, offset_y


class BaseEvaluator(nn.Module):
    def __init__(self, left_x, left_y, group_size):
        super().__init__()
        self.left_x = left_x
        self.left_y = left_y
        self.group_size = group_size

    def _normalize_coords(self, coords, dim_size):
        coords = coords - (dim_size - 1) / 2
        coords = coords / ((dim_size - 1) / 2)
        return coords

    def forward(self, features, offset_x, offset_y, roll_steps):
        b, c, h, w = features.shape
        device = offset_x.device

        left_x = self.left_x.to(device)[:b]
        left_y = self.left_y.to(device)[:b]

        right_x = torch.clamp(left_x + offset_x, 0, w - 1)
        right_y = torch.clamp(left_y + offset_y, 0, h - 1)

        right_x = self._normalize_coords(right_x, w)
        right_y = self._normalize_coords(right_y, h)

        samples = torch.cat((right_x.unsqueeze(4), right_y.unsqueeze(4)), dim=4)

        offset_strength_list = []

        for i in range(samples.shape[1]):
            grid = samples[:, i]
            warped = F.grid_sample(features, grid.half(), align_corners=True)
            warped2 = torch.roll(warped, roll_steps[0], 1)
            warped3 = torch.roll(warped, roll_steps[1], 1)

            def mean_strength(w):
                return torch.cat([
                    torch.mean(w[:, j:j + self.group_size, :, :], dim=1, keepdim=True)
                    for j in range(0, w.shape[1], self.group_size)
                ], dim=1)

            offset_strength = torch.cat([
                mean_strength(-torch.abs(features - warped)),
                mean_strength(-torch.abs(features - warped2)),
                mean_strength(-torch.abs(features - warped3))
            ], dim=1)

            offset_strength = offset_strength.sort(dim=1, descending=True).values[:, 0:1]
            offset_strength_list.append(offset_strength)

        offset_strength = torch.cat(offset_strength_list, dim=1)
        offset_strength = F.softmax(offset_strength * 1000, dim=1)

        weighted_x = torch.sum(offset_x * offset_strength, dim=1, keepdim=True)
        weighted_y = torch.sum(offset_y * offset_strength, dim=1, keepdim=True)

        final_x = torch.clamp(weighted_x + left_x, 0, w - 1) - left_x
        final_y = torch.clamp(weighted_y + left_y, 0, h - 1) - left_y

        return final_x, final_y


class EvaluateZM(BaseEvaluator):
    def __init__(self, left_x, left_y):
        super().__init__(left_x, left_y, group_size=12)

    def forward(self, features, offset_x, offset_y):
        return super().forward(features, offset_x, offset_y, roll_steps=(12, 24))


class EvaluateCNN(BaseEvaluator):
    def __init__(self, left_x, left_y):
        super().__init__(left_x, left_y, group_size=32)

    def forward(self, features, offset_x, offset_y):
        return super().forward(features, offset_x, offset_y, roll_steps=(32, 64))


class PatchMatch(nn.Module):
    def __init__(self, iteration_count, random_search_window_size,size,batch_size):
        """
        PatchMatch algorithm for correspondence estimation using two feature maps.

        Args:
            iteration_count (int): Number of iterations for refinement.
            random_search_window_size (int): Search window size for random sampling.
            size (int): Spatial size of the input feature maps.
            batch_size (int): Expected batch size.
        """
        super(PatchMatch, self).__init__()
        self.batch_size = batch_size
        self.size = size
        self.iteration_count = iteration_count
        self.window_size = random_search_window_size

        # Random sampler module
        self.random_sample     = RandomSampler()

        # Create and expand mesh grid coordinates for the batch
        self.left_x_coordinate = torch.arange(0.0, size).repeat(size).view(size, size)
        self.left_y_coordinate = torch.arange(0.0, size).unsqueeze(1).repeat(1, size).view(size, size)
        self.left_x_coordinate = torch.clamp(self.left_x_coordinate, min=0, max=size - 1)
        self.left_y_coordinate = torch.clamp(self.left_y_coordinate, min=0, max=size - 1)
        self.left_x_coordinate = self.left_x_coordinate.expand(batch_size, -1, -1).unsqueeze(1) 
        self.left_y_coordinate = self.left_y_coordinate.expand(batch_size, -1, -1).unsqueeze(1)

        # Initialize evaluation modules
        self.evaluate_ZM  = EvaluateZM(self.left_x_coordinate, self.left_y_coordinate)
        self.evaluate_CNN = EvaluateCNN(self.left_x_coordinate, self.left_y_coordinate)

        # Offsets
        self.min_offset_x = -self.left_x_coordinate
        self.min_offset_y = -self.left_y_coordinate
        self.max_offset_x = self.size - self.left_x_coordinate - 1
        self.max_offset_y = self.size - self.left_y_coordinate - 1

    @staticmethod
    def propagation(offset_x, offset_y):
        """
        Propagate offset tensors using direct neighbors and extrapolated positions.
        """
        def propagate_offset(offset):
            return torch.cat((
                offset,
                # Direct Neighbors
                torch.roll(offset,  1, 3),
                torch.roll(offset, -1, 3),
                torch.roll(offset,  1, 2),
                torch.roll(offset, -1, 2),
                # Extrapolated Neighbors
                (2 * torch.roll(offset,  1, 3)) - torch.roll(offset,  2, 3),
                (2 * torch.roll(offset, -1, 3)) - torch.roll(offset, -2, 3),
                (2 * torch.roll(offset,  1, 2)) - torch.roll(offset,  2, 2),
                (2 * torch.roll(offset, -1, 2)) - torch.roll(offset, -2, 2),
                (2 * torch.roll(offset, shifts=( 1,  1), dims=(2, 3))) - torch.roll(offset, shifts=( 2,  2), dims=(2, 3)),
                (2 * torch.roll(offset, shifts=(-1, -1), dims=(2, 3))) - torch.roll(offset, shifts=(-2, -2), dims=(2, 3)),
                (2 * torch.roll(offset, shifts=( 1, -1), dims=(2, 3))) - torch.roll(offset, shifts=( 2, -2), dims=(2, 3)),
                (2 * torch.roll(offset, shifts=(-1,  1), dims=(2, 3))) - torch.roll(offset, shifts=(-2,  2), dims=(2, 3))
            ), dim=1)
        return propagate_offset(offset_x), propagate_offset(offset_y)

    def fix_out_of_coordinate(self, offset_x, offset_y):
        """Wrap offsets to keep them within bounds."""
        left_y = self.left_y_coordinate.to(offset_y.device)[:offset_y.size(0)]
        left_x = self.left_x_coordinate.to(offset_x.device)[:offset_x.size(0)]
        offset_y = torch.remainder(offset_y + left_y, self.size) - left_y
        offset_x = torch.remainder(offset_x + left_x, self.size) - left_x
        return offset_x, offset_y

    def non_local(self, offset_x, offset_y, threshold):
        """
        Applies non-local perturbation to offset_x and offset_y based on a distance threshold.
        """
        batch_size = offset_x.size(0)

        # Ensure min_offset tensors are on the same device as offsets
        min_offset_x = self.min_offset_x.to(offset_x.device)
        min_offset_y = self.min_offset_y.to(offset_y.device)

        # Trim min_offset tensors if batch size is smaller
        if batch_size != min_offset_x.size(0):
            min_offset_x = min_offset_x[:batch_size]
            min_offset_y = min_offset_y[:batch_size]

        # Compute distance limit mask
        limit_mask = (offset_x ** 2 + offset_y ** 2) <= threshold

        # Generate random displacements within size range
        offset_x1 = min_offset_x + self.size * torch.rand(min_offset_x.size(), device=offset_x.device)
        offset_y1 = min_offset_y + self.size * torch.rand(min_offset_y.size(), device=offset_y.device)

        # Apply non-local replacements conditionally
        offset_x = torch.where(limit_mask, offset_x1, offset_x)
        offset_y = torch.where(limit_mask, offset_y1, offset_y)
        return offset_x, offset_y

    def random_search_window(self, offset_x, offset_y):
        """Generate local search window bounds for random sampling."""
        half = self.window_size // 2
        return offset_x - half, offset_x + half, offset_y - half, offset_y + half

    def forward(self, ZM_features, CNN_features, CNN_features_g):
        device = CNN_features.get_device()
        
        left_x_coordinate = self.left_x_coordinate.to(device)
        left_y_coordinate = self.left_y_coordinate.to(device)

        min_offset_x_zm   = self.min_offset_x.to(device)
        max_offset_x_zm   = self.max_offset_x.to(device)
        min_offset_y_zm   = self.min_offset_y.to(device)
        max_offset_y_zm   = self.max_offset_y.to(device)
        min_offset_x_cnn  = self.min_offset_x.to(device)
        max_offset_x_cnn  = self.max_offset_x.to(device)
        min_offset_y_cnn  = self.min_offset_y.to(device)
        max_offset_y_cnn  = self.max_offset_y.to(device)

        if CNN_features.size()[0] != self.batch_size:
            left_x_coordinate = left_x_coordinate[:CNN_features.size()[0],:,:,:]
            left_y_coordinate = left_y_coordinate[:CNN_features.size()[0], :, :, :]
            min_offset_x_zm   = min_offset_x_zm[:CNN_features.size()[0], :, :, :]
            max_offset_x_zm   = max_offset_x_zm[:CNN_features.size()[0], :, :, :]
            min_offset_y_zm   = min_offset_y_zm[:CNN_features.size()[0], :, :, :]
            max_offset_y_zm   = max_offset_y_zm[:CNN_features.size()[0], :, :, :]
            min_offset_x_cnn  = min_offset_x_cnn[:CNN_features.size()[0], :, :, :]
            max_offset_x_cnn  = max_offset_x_cnn[:CNN_features.size()[0], :, :, :]
            min_offset_y_cnn  = min_offset_y_cnn[:CNN_features.size()[0], :, :, :]
            max_offset_y_cnn  = max_offset_y_cnn[:CNN_features.size()[0], :, :, :]

        offset_x_zm  = min_offset_x_zm  + self.size * torch.rand(min_offset_x_zm.size(), device=device)
        offset_y_zm  = min_offset_y_zm  + self.size * torch.rand(min_offset_x_zm.size(), device=device)
        offset_x_cnn = min_offset_x_cnn + self.size * torch.rand(min_offset_x_zm.size(), device=device)
        offset_y_cnn = min_offset_y_cnn + self.size * torch.rand(min_offset_x_zm.size(), device=device)

        for iter_num in range(self.iteration_count):
            offset_x_zm, offset_y_zm = self.random_sample(min_offset_x_zm, max_offset_x_zm, 
                                                          min_offset_y_zm, max_offset_y_zm, 
                                                          offset_x_zm, offset_y_zm)

            offset_x_zm, offset_y_zm   = self.fix_out_of_coordinate(offset_x_zm, offset_y_zm)
            offset_x_zm, offset_y_zm   = self.evaluate_ZM(ZM_features, offset_x_zm, offset_y_zm)

            offset_x_cnn, offset_y_cnn = self.random_sample(min_offset_x_cnn, max_offset_x_cnn, 
                                                            min_offset_y_cnn, max_offset_y_cnn, 
                                                            offset_x_cnn, offset_y_cnn)
            offset_x_cnn, offset_y_cnn = self.fix_out_of_coordinate(offset_x_cnn, offset_y_cnn)
            offset_x_cnn, offset_y_cnn = self.evaluate_CNN(CNN_features, offset_x_cnn, offset_y_cnn)

            if iter_num < self.iteration_count-1:
                offset_x_zm, offset_y_zm   = self.propagation(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm   = self.fix_out_of_coordinate(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm   = self.evaluate_ZM(ZM_features, offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm   = self.propagation(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm   = self.fix_out_of_coordinate(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm   = self.evaluate_ZM(ZM_features, offset_x_zm, offset_y_zm)

                offset_x_cnn, offset_y_cnn = self.propagation(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.fix_out_of_coordinate(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.evaluate_CNN(CNN_features, offset_x_cnn, offset_y_cnn)

                offset_x_zm, offset_y_zm   = self.non_local(offset_x_zm, offset_y_zm, 25)
                offset_x_cnn, offset_y_cnn = self.non_local(offset_x_cnn, offset_y_cnn, 25)

                search_window_zm  = self.random_search_window(offset_x_zm, offset_y_zm)
                search_window_cnn = self.random_search_window(offset_x_cnn, offset_y_cnn)
                min_offset_x_zm, max_offset_x_zm, min_offset_y_zm, max_offset_y_zm     = search_window_zm
                min_offset_x_cnn, max_offset_x_cnn, min_offset_y_cnn, max_offset_y_cnn = search_window_cnn
            else:
                offset_x_cnn, offset_y_cnn = self.propagation(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.fix_out_of_coordinate(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.evaluate_CNN(CNN_features_g, offset_x_cnn, offset_y_cnn)

        x_zm  = left_x_coordinate + offset_x_zm
        y_zm  = left_y_coordinate + offset_y_zm
        x_cnn = left_x_coordinate + offset_x_cnn
        y_cnn = left_y_coordinate + offset_y_cnn

        coordinates  = [x_zm, y_zm, x_cnn, y_cnn] 
        offsets      = [offset_x_zm, offset_y_zm, offset_x_cnn, offset_y_cnn]

        return coordinates, offsets

