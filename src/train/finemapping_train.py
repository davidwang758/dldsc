import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ..model.finemapping_loss import susie_elbo
from ..model.finemapping_loss import finemap_elbo
from ..model.finemapping_loss import finemap_inf_elbo

class FinemappingTrainer():
    def __init__(self, z , R, N, model, lr, n_epoch, loss_f):
        super().__init__()
        self.device = z.device
        self.z = z
        self.R = R
        self.N = torch.tensor(N).to(self.device)
        self.p = len(z)

        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.lr = lr
        self.n_epoch = n_epoch
        self.loss=[]
        self.kl1=[]
        self.kl2=[]
        self.nlk=[]
    
        #self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.opt = optim.LBFGS(self.model.parameters(), lr=lr)
        self.sch = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=n_epoch, eta_min=1e-6)
        #self.sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, T_0=500, eta_min=1e-7)

    def closure(self):
            self.opt.zero_grad()
            mu, s2, alpha = self.model()
            elbo_loss, nlk, kl1, kl2 = self.loss_f(self.z, self.R, self.N, mu, s2, self.tau2, alpha, self.pi)            
            elbo_loss.backward()
            return elbo_loss
    
    def train(self):
        self.model.train()
        for i in range(self.n_epoch):  
            elbo_loss = self.opt.step(self.closure)
            self.sch.step()

            #if i % 500 == 0:
            print(f"Elbo loss: {elbo_loss.item()}")
            self.loss.append(elbo_loss.item())

class SusieModel(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros((k,m)))
        self.s2 = torch.nn.Parameter(torch.ones((k,m)) * -10)
        self.alpha = torch.nn.Parameter(torch.ones((k,m)) * -5)

    def forward(self):
        mu2 = F.softplus(self.s2) + self.mu**2
        return self.mu, mu2, F.softmax(self.alpha, dim=-1)
    
class FinemapModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(m))
        self.s2 = torch.nn.Parameter(torch.ones(m) * -10)
        self.alpha = torch.nn.Parameter(torch.ones(m) * -4.0)

    def forward(self):
        return self.mu, F.softplus(self.s2) + 1e-10, torch.sigmoid(self.alpha) + 1e-10
    
class FinemapInfModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.mu1 = torch.nn.Parameter(torch.zeros(m))
        self.mu0 = torch.nn.Parameter(torch.zeros(m))
        self.s21 = torch.nn.Parameter(torch.ones(m) * -5.0)
        self.s20 = torch.nn.Parameter(torch.ones(m) * -5.0)
        self.alpha = torch.nn.Parameter(torch.ones(m) * -4.0)

    def forward(self):
        return self.mu1, self.mu0, F.softplus(self.s21), F.softplus(self.s20), torch.sigmoid(self.alpha)
        
class Susie(FinemappingTrainer):
    def __init__(self, z, R, N, k, lr=1e-2, n_epoch=1000):
        self.model = SusieModel(len(z), k)

        super().__init__(z, R, N, self.model, lr, n_epoch, susie_elbo)
        self.k = k

        self.pi = torch.ones((self.k, self.p)).to(self.device) * 1/self.p
        self.tau2 = torch.ones((self.k, self.p)).to(self.device) * 0.0003215041

class Finemap(FinemappingTrainer):
    def __init__(self, z, R, N, lr=1e-2, n_epoch=1000):
        self.model = FinemapModel(len(z))

        super().__init__(z, R, N, self.model, lr, n_epoch, finemap_elbo)
        self.pi = torch.ones(self.p).to(self.device) * 1/self.p
        self.tau2 = torch.ones(self.p).to(self.device) * 1e-3

class FinemapInf(FinemappingTrainer):
    def __init__(self, z, R, N, lr=1e-2, n_epoch=1000):
        self.model = FinemapInfModel(len(z))

        super().__init__(z, R, N, self.model, lr, n_epoch, finemap_inf_elbo)
        self.pi = torch.ones(self.p).to(self.device) * 1/self.p
        self.tau2 = torch.ones(self.p).to(self.device) * 1e-2
        self.sigma20 = torch.ones(self.p).to(self.device) * 1e-8
        

    def closure(self):
            self.opt.zero_grad()
            mu1, mu0, s21, s20, alpha = self.model()
            elbo_loss, nlk, kl1, kl2 = self.loss_f(self.z, self.R, self.N, mu1, s21, mu0, s20, self.tau2, self.sigma20, alpha, self.pi)            
            elbo_loss.backward()
            return elbo_loss

        
