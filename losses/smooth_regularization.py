import torch
import torch.nn as nn

class SmoothLoss(nn.Module):
    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        diff = frame_feats[:, 1:, :] - frame_feats[:, :-1, :]
        return diff.square().sum(dim=-1).mean()
