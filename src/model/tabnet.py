import torch.nn as nn
import torch
from pytorch_tabnet.tab_network import TabNetNoEmbeddings

class DLDSCTabNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        output_activation="softplus",
        device="cuda"
    ):
        super(DLDSCTabNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type

        self.sigma2 = nn.Parameter(torch.ones(1, output_dim))
        if output_activation == "softplus":
            self.output_activation = nn.functional.softplus
        elif output_activation == "softmax": 
            self.output_activation = lambda x: F.softmax(x, dim=0)
        else:
            self.output_activation = lambda x: x

        self.tabnet = TabNetNoEmbeddings(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=torch.eye(self.input_dim).to(device)
        )

    def forward(self, x):
        preds, M_loss = self.tabnet(x)
        preds = self.output_activation(preds)
        return preds, M_loss

    def forward_masks(self, x):
        return self.tabnet.forward_masks(x)