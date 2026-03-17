import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli, kl_divergence
import pandas as pd
from . import mlp

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x, temperature=0.5):
        logit = self.network(x)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logit)))
        noisy_logit = (logit + gumbel_noise) / temperature
        soft_sample = torch.sigmoid(noisy_logit)

        return soft_sample
    
class Elbo_Loss(nn.Module):
    def __init__(self, max_k, R, z):
        super(Elbo_Loss, self).__init__()
        self.max_k = max_k 
        self.S = torch.matmul(torch.inverse(R), z)

    def forward(self, R, z, N, sigma2, p, pq, gamma):
        # Find non-zero locations in indicator vector gamma
        (u, ind) = torch.topk(gamma, self.max_k)
        ind = ind[torch.where(u > 0.01)[0]]
        k = len(ind)

        # Compute loss
        loss = 0

        # Compute NLL of MVN using Woodbury identity for inverses
        U =  N * torch.diag(sigma2 * gamma)[:,ind]
        V = R[ind,:]

        inv = torch.inverse(torch.eye(k) + torch.mm(V,U))
        sigma_inv = torch.eye(len(z)) - torch.mm(torch.mm(U,inv),V)
        sigma = torch.eye(k) + torch.mm(V,U)
        quadratic_form = -torch.matmul(torch.matmul(z.T, sigma_inv),self.S)/2
        log_likelihood =  -torch.logdet(sigma)/2 + quadratic_form # Does this need the prior?

        # Compute KLD regularization   
        x2 = pq[ind] # Why filter out the zero indexes?
        x1 = p[ind]
        s1 = torch.sum(x2 * (torch.log(x2) - torch.log(x1)))
        s2 = torch.sum((1 - x2) * (torch.log(1 - x2) - torch.log(1 - x1)))  
        kl_loss = s1 + s2
             
        loss += -log_likelihood + kl_loss
  
        return loss

def get_finetuning_model(model_class, mode):
    assert mode in ["Susie", "Finemap"], "Mode must be Susie or Finemap"
    assert model_class in ["MLP_Baseline", "MLP_Borzoi"], "Model class must be MLP_Baseline or MLP_Borzoi"

    model_class = getattr(mlp, model_class)

    class MLP_Finemapping(model_class):
        def __init__(self, input_dim, p=0.0, trait_index=None):
            super(MLP_Finemapping, self).__init__(input_dim, 1, p=p)
            del self.sigma2

            layer_list = list(self.layers.children())
            self.layers = torch.nn.Sequential(*layer_list[:-1])
            self.output = layer_list[-1]
            self.trait_index = trait_index
            
            with torch.no_grad():
                torch.nn.init.zeros_(self.output.weight)
                torch.nn.init.constant_(self.output.bias, -5.0)

            if mode == "Susie":
                self.transform = lambda x: F.softmax(x, dim=0)
            else:
                self.transform = torch.sigmoid 

        def load_state_dict(self, state_dict, strict=True):         
            if 'output.weight' not in state_dict:
                state_dict = state_dict.copy()
                old_w_key = f'layers.{len(self.layers)}.weight'
                old_b_key = f'layers.{len(self.layers)}.bias'
                multitask_w = state_dict.pop(old_w_key)
                multitask_b = state_dict.pop(old_b_key)
                state_dict['output.weight'] = multitask_w[self.trait_index].unsqueeze(0)
                state_dict['output.bias'] = multitask_b[self.trait_index].unsqueeze(0)
                state_dict.pop('sigma2')

            return super().load_state_dict(state_dict, strict=strict)
                
        def forward(self, x):
            return self.transform(self.output(self.layers(x)))
        
    return MLP_Finemapping
    
import collections

def fix_old_keys(state_dict):
    new_state_dict = collections.OrderedDict()

    for key, value in state_dict.items():
        new_key = key
        
        # Replace 'features' with 'layers'
        if "features" in new_key:
            new_key = new_key.replace("features", "layers")
        
        # Replace 'classifier' with 'output'
        if "classifier" in new_key:
            new_key = new_key.replace("classifier", "output")
            
        new_state_dict[new_key] = value

    return new_state_dict

class Finemapping_Inference(nn.Module):
    def __init__(self, input_dim, model_file, model_class, mode, trait_index=None, device="cuda"):
        super(Finemapping_Inference, self).__init__()
        self.model_dict = nn.ModuleDict()
        
        for idx, row in model_file.iterrows():
            model_path = row['MODEL']
            chrom = row['CHR']
            
            ModelClass = get_finetuning_model(model_class, mode)
            model = ModelClass(input_dim, p=0.0, trait_index=trait_index).to(device) # Set dropout to 0 for inference
            checkpoint = torch.load(model_path, map_location=device)
            checkpoint = fix_old_keys(checkpoint) # fix the old keys for backward compatibility
            model.load_state_dict(checkpoint)
            
            keys = str(chrom).split(",")
            for k in keys:
                self.model_dict[k] = model

    def forward(self, x, chr):
        return self.model_dict[chr](x)