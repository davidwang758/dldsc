import torch
import math
# Susie Sufficient Statistics

# Gaussian KLD
def _kl_beta_susie(mu, mu2, tau2, alpha):
    s2 = mu2 - mu**2
    kl_beta = 0.5 * ((mu**2 + s2) / tau2 - torch.log(s2) + torch.log(tau2) - 1)
    return torch.sum(alpha * kl_beta) # Sum over k x m matrix

# Multinomial KLD
def _kl_gamma_susie(alpha, pi, eps=1e-10):
    kl_gamma = torch.log(alpha + eps) - torch.log(pi + eps)
    return torch.sum(alpha * kl_gamma) # Sum over k x m matrix

# Expectation of loglikelihood
def _nloglik_susie(xtx, xty, mu, mu2, alpha):
    mu = alpha * mu # First moment
    mu_bar = torch.sum(mu, axis=0) # Average first moment
    mu2 = alpha * mu2 # Second moment

    var_term = torch.sum(mu2 * torch.diag(xtx).unsqueeze(0)) - torch.sum((mu @ xtx) * mu) #  Tr(X^TX * Cov[B]) 
    term1 = -2 * torch.sum(xty * mu_bar) #  -2X^TYE[B]
    term2 = torch.sum(mu_bar * (xtx @ mu_bar)) + var_term  # E[B]X^TXE[B]
 
    loss =  0.5 * (term1 + term2)
    
    return loss

# z: z-scores 
# R: LD matrix 
# N: sample size
# mu: posterior first moment (mean)
# mu2: posterior second moment (variance = second moment - first moment^2)
# tau2: prior variance
# alpha: posterior inclusion probability
# pi: prior inclusion priobability
# sigma2: residual variance
# mu, s2 and alpha are free paramters to be fit (should be predicted by encoder)
# tau2 and pi are priors which are fixed constants (can be predicted from annotation)
def susie_elbo(z, R, N, mu, mu2, tau2, alpha, pi):
    xtx = R * N
    xty = z * math.sqrt(N) # z might be a variant by trait matrix.
    nlk = _nloglik_susie(xtx, xty, mu, mu2, alpha)
    kl1 = _kl_beta_susie(mu, mu2, tau2, alpha)
    kl2 = _kl_gamma_susie(alpha, pi)
        
    return nlk + kl1 + kl2, nlk, kl1, kl2

def susie_elbo_lr(qtw, qtq, mu, mu2, tau2, alpha, pi):
    nlk = _nloglik_susie(qtq, qtw, mu, mu2, alpha)
    kl1 = _kl_beta_susie(mu, mu2, tau2, alpha)
    kl2 = _kl_gamma_susie(alpha, pi)

    return nlk + kl1 + kl2, nlk, kl1, kl2

# Susie Multitask

# Gaussian KLD
def _kl_beta_susie_mt(mu, mu2, tau2, alpha):
    tau2 = tau2.view(-1,1,1)
    s2 = mu2 - mu**2
    kl_beta = 0.5 * ((mu**2 + s2) / tau2  - torch.log(s2) + torch.log(tau2) - 1)
    return torch.sum(alpha * kl_beta) # Sum over k x m matrix

# Multinomial KLD
def _kl_gamma_susie_mt(alpha, pi, eps=1e-10):
    kl_gamma = torch.log(alpha + eps) - torch.log(pi.unsqueeze(0) + eps)
    return torch.sum(alpha * kl_gamma) # Sum over k x m matrix

# Expectation of loglikelihood
def _nloglik_susie_mt(xtx, xty, mu, mu2, alpha):
    mu = alpha * mu # First moment
    mu_bar = torch.sum(mu, axis=0) # Average first moment
    mu2 = alpha * mu2 # Second moment

    var_term = torch.sum(mu2 * torch.diag(xtx).view(1,-1,1)) - torch.sum((torch.einsum('kiv, ij -> kjv', mu, xtx)) * mu) #  Tr(X^TX * Cov[B]) 
    term1 = -2 * torch.sum(xty * mu_bar) #  -2X^TYE[B]
    term2 = torch.sum(mu_bar * (xtx @ mu_bar)) + var_term  # E[B]X^TXE[B]
 
    loss =  0.5 * (term1 + term2)
    
    return loss

# z: z-scores 
# R: LD matrix 
# N: sample size
# mu: posterior first moment (mean)
# mu2: posterior second moment (variance = second moment - first moment^2)
# tau2: prior variance
# alpha: posterior inclusion probability
# pi: prior inclusion priobability
# sigma2: residual variance
# mu, s2 and alpha are free paramters to be fit (should be predicted by encoder)
# tau2 and pi are priors which are fixed constants (can be predicted from annotation)
# z is variant x view
# R is variant x variant
# mu, mu2, and alpha are L x variant x view.
# pi is variant x view.
# Tau is a scalar (can we preselect a grid?).
def multitask_susie_elbo(z, R, N, mu, mu2, tau2, alpha, pi):
    xtx = R * N
    xty = z * math.sqrt(N) # z might be a variant by trait matrix.
    nlk = _nloglik_susie_mt(xtx, xty, mu, mu2, alpha)
    kl1 = _kl_beta_susie_mt(mu, mu2, tau2, alpha)
    kl2 = _kl_gamma_susie_mt(alpha, pi)
        
    return nlk + kl1 + kl2, nlk, kl1, kl2

# FINEMAP Sufficient Statistics

# Gaussian KLD
def _kl_beta_finemap(mu, s2, tau2, alpha):
    kl_beta = 0.5 * ((mu**2 + s2) / tau2 - torch.log(s2) + torch.log(tau2) - 1)
    return torch.sum(alpha * kl_beta)

# Bernoulli KLD
def _kl_gamma_finemap(alpha, pi, eps=1e-10):
    term1 = torch.sum((alpha + eps) * (torch.log(alpha + eps) - torch.log(pi + eps)))
    term2 = torch.sum((1 - alpha + eps) * (torch.log(1 - alpha + eps) - torch.log(1 - pi + eps)))
    return term1 + term2

# Expectation of loglikelihood
def _nloglik_finemap(xtx, xty, mu, s2, alpha):
    mu_bar = alpha * mu
    s2_bar = alpha * s2 + alpha * (1 - alpha) * mu**2 

    term1 = -2 * torch.sum(xty * mu_bar) # -2X^TYE[B]
    term2 = torch.sum(mu_bar * (xtx @ mu_bar)) + torch.sum(torch.diag(xtx) * s2_bar) # E[B]X^TXE[B] + Tr(X^TX * Cov[B]) 
    
    loss = 0.5 * (term1 + term2)
    
    return loss

# z: z-scores 
# R: LD matrix 
# N: sample size
# mu: posterior mean
# s2: posterior variance
# tau2: prior variance
# alpha: posterior inclusion probability
# pi: prior inclusion priobability
# sigma2: residual variance
# mu, s2 and alpha are free paramters to be fit (should be predicted by encoder)
# tau2 and pi are priors which are fixed constants (can be predicted from annotation)
# sigma2 is the residual variance and fixed at 1 for continous traits
def finemap_elbo(z, R, N, mu, s2, tau2, alpha, pi):
    xtx = R * N
    xty = z * math.sqrt(N)
    nlk = _nloglik_finemap(xtx, xty, mu, s2, alpha)
    kl1 = _kl_beta_finemap(mu, s2, tau2, alpha)
    kl2 = _kl_gamma_finemap(alpha, pi)
    return nlk + kl1 + kl2, nlk, kl1, kl2

# FINEMAP Low Rank

def finemap_elbo_lr(qtw, qtq, mu, s2, tau2, alpha, pi):
    nlk = _nloglik_finemap(qtq, qtw, mu, s2, alpha)
    kl1 = _kl_beta_finemap(mu, s2, tau2, alpha)
    kl2 = _kl_gamma_finemap(alpha, pi)
    
    return nlk + kl1 + kl2, nlk, kl1, kl2

# FINEMAP-INF Sufficient Statistics

# Gaussian KLD
def _kl_beta_finemap_inf(mu1, s21, mu0, s20, tau2, sigma20, alpha):
    kl_beta1 = (mu1**2 + s21) / (tau2 + sigma20) - torch.log(s21) + torch.log(tau2 + sigma20) - 1
    kl_beta0 = (mu0**2 + s20) / sigma20 - torch.log(s20) + torch.log(sigma20) - 1
    return 0.5 * torch.sum(alpha * kl_beta1 + (1-alpha) * kl_beta0)

# Expectation of loglikelihood
def _nloglik_finemap_inf(xtx, xty, mu1, s21, mu0, s20, alpha):
    mu_bar = alpha * mu1 + (1 - alpha) * mu0
    s2_bar = alpha * s21 + (1 - alpha) * s20 + alpha * (1 - alpha) * (mu1 - mu0)**2 

    term1 = -2 * torch.sum(xty * mu_bar) # -2X^TYE[B]
    term2 = torch.sum(mu_bar * (xtx @ mu_bar)) + torch.sum(torch.diag(xtx) * s2_bar) # E[B]X^TXE[B] + Tr(X^TX * Cov[B]) 
    
    loss = 0.5 * (term1 + term2)
    
    return loss

def finemap_inf_elbo(z, R, N, mu1, s21, mu0, s20, tau2, sigma20, alpha, pi):
    xtx = R * N
    xty = z * math.sqrt(N)
    nlk = _nloglik_finemap_inf(xtx, xty, mu1, s21, mu0, s20, alpha)
    kl1 = _kl_beta_finemap_inf(mu1, s21, mu0, s20, tau2, sigma20, alpha)
    kl2 = _kl_gamma_finemap(alpha, pi)
    total_loss = nlk + kl1 + kl2
    return total_loss, nlk, kl1, kl2