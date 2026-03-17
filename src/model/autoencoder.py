import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.0):
        super(Autoencoder, self).__init__()
        # Encoder: Compresses the input
        self.encoder = nn.Sequential(
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
            nn.Dropout(p)
        )
        # Decoder: Reconstructs the input from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4096, output_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded