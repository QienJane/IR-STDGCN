import torch
import torch.nn as nn

class SmoothLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        diff = frame_feats[:, 1:, :] - frame_feats[:, :-1, :]
        smooth_loss = (diff * diff).mean()
  
        return self.lambda_smooth * smooth_loss