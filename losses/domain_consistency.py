import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    """Domain consistency loss L_DC (Eq. (4)) on the new class logits of the real samples.

    scores: [n x n_new] logits of the classes added in the current task.
    binary=True (one new class): binary cross-entropy of the Sigmoid output, target_scores [n x 1].
    binary=False (several new classes): Softmax cross-entropy over the new classes, target_scores [n]
    are the labels counted from the first new class.
    data_weights: class-balance weights of the samples, applied before the mean.
    """

    def forward(self, scores, target_scores, data_weights=None, binary=False):
        if binary:
            loss = F.binary_cross_entropy_with_logits(scores, target_scores, reduction='none')
            loss = loss.reshape(scores.size(0), -1).mean(dim=1)
        else:
            loss = F.cross_entropy(scores, target_scores.long(), reduction='none')
        if data_weights is not None:
            weights = torch.as_tensor(data_weights, device=loss.device).reshape(-1)
            loss = loss * weights
        return loss.mean()
