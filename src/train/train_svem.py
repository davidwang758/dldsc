import torch.optim as optim
import torch 
from ..model.finemapping_loss import finemap_elbo
from  ..train.cavi import finemap_cavi
from ..model.mlp import MLP_Baseline

class MLP_Finetune(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(MLP_Finetune, self).__init__()

        original_layers = list(pretrained_model.layers.children())
        self.features = torch.nn.Sequential(*original_layers[:-1])
        self.classifier = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

def get_model(model_checkpoint, device="cuda"):
    pretrained_model = MLP_Baseline(187, 49).to(device)
    state_dict = torch.load(model_checkpoint, map_location=device)
    pretrained_model.load_state_dict(state_dict)
    model = MLP_Finetune(pretrained_model).to(device)
    with torch.no_grad():
        torch.nn.init.zeros_(model.classifier.weight)
        torch.nn.init.constant_(model.classifier.bias, -5.0)
    return model


# Eventually replace z_list and R_list with the D-LDSC dataloader.
def train_svem(z_list, R_list, N, A_list, model, lr=1e-2, n_epoch=100, device="cuda"):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=n_epoch,
                                                     eta_min=1e-6)
    
    model.train()
    loss = []
    mu_init = [None]*len(z_list)
    alpha_init = [None]*len(z_list)
    mu = None
    s2 = None
    alpha = None
    pi = None
    for i in range(n_epoch):
        epoch_elbo_loss = 0.0
        j = 0
        for z, R, A in zip(z_list, R_list, A_list): 
            # Fix prior variance
            tau2 = torch.ones(len(z)).to(device) * 1e-3

            # Predict prior causal probability
            pi = model(A).squeeze(1)

            # Fit posterior to convergence (E step)
            with torch.no_grad():
                mu, s2, alpha = finemap_cavi(z, R, N, tau2, pi, mu_init[j], alpha_init[j], max_iter=100)
                mu_init[j] = mu
                alpha_init[j] = alpha
 
            # M step
            elbo_loss, nlk, kl1, kl2 = finemap_elbo(z, R, N, mu, s2, tau2, alpha, pi) 
            optimizer.zero_grad()
            elbo_loss.backward()
            optimizer.step()

            epoch_elbo_loss += elbo_loss.item()
            j +=1 
            
        scheduler.step()
        loss.append(epoch_elbo_loss)
                
        print(f"Elbo loss: {elbo_loss.item()}")

    return model, loss, mu, s2, alpha, pi

 