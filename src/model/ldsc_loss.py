import torch.nn as nn
import torch

class LDSC_Loss(nn.Module):
    def __init__(self, eps=None):
        super(LDSC_Loss, self).__init__()
        self.eps = eps

    def forward(self, yhat, y, R2, w, sigma2, trait_specific=False):
        if self.eps is not None: 
            R2 = R2 + torch.eye(R2.size(0), device=R2.device, dtype=R2.dtype) * self.eps
        yhat = R2 @ yhat + sigma2
        loss = (y - yhat)**2 * w
        if trait_specific:
            return torch.mean(loss), torch.mean(loss, axis=0)
        else:
            return torch.mean(loss)

class SLDSC_Loss_No_R2(nn.Module):
    def __init__(self):
        super(SLDSC_Loss_No_R2, self).__init__()
    
    def forward(self, yhat, y, R2, w, sigma2):
        # Don't use R2
        yhat = yhat + sigma2
        return torch.mean((y - yhat)**2 * w)