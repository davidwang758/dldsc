import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP_Baseline(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.0, output_activation="softplus"):
        super(MLP_Baseline, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(16, output_dim)
        )
        self.sigma2 = nn.Parameter(torch.ones(1, output_dim))
        if output_activation == "softplus":
            self.output_activation = nn.functional.softplus
        elif output_activation == "softmax":
            self.output_activation = lambda x: F.softmax(x, dim=0)
        else:
            self.output_activation = lambda x: x

    def forward(self, x):
        yhat = self.layers(x)   
        yhat = self.output_activation(yhat)
        return yhat

class MLP_Borzoi(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.0, output_activation="softplus"):
        super(MLP_Borzoi, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(16, output_dim)
        )
        self.sigma2 = nn.Parameter(torch.ones(1, output_dim))
        if output_activation == "softplus":
            self.output_activation = nn.functional.softplus
        elif output_activation == "softmax":
            self.output_activation = lambda x: F.softmax(x, dim=0)
        else:
            self.output_activation = lambda x: x

    def forward(self, x):
        yhat = self.layers(x)   
        yhat = self.output_activation(yhat)
        return yhat