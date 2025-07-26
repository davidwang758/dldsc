import torch.nn as nn
import torch

class SLDSC_Loss(nn.Module):
    def __init__(self, eps=None):
        super(SLDSC_Loss, self).__init__()
        self.eps = eps

    def forward(self, yhat, y, R2, w, sigma2):
        if self.eps is not None: 
            R2 = R2 + torch.eye(R2.size(0), device=R2.device, dtype=R2.dtype) * self.eps
        yhat = R2 @ yhat + sigma2
        return torch.mean((y - yhat)**2 * w)