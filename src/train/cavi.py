import torch
import math

def finemap_cavi(z, R, N, tau2, pi, mu_init=None, alpha_init=None, max_iter=100, delta=1e-4, sigma2=1, device="cuda"):
    # compute sufficient statistics
    xty = math.sqrt(N) * z
    xtx = N * R

    # precompute
    m = len(xty)
    s2 = 1.0 / (1.0 / tau2 + torch.diag(xtx) / sigma2)
    alpha_const = torch.log(pi) - torch.log(1.0 - pi) + 0.5 * (torch.log(s2) - torch.log(tau2))

    # initialization
    if mu_init is None:
        mu = torch.zeros(m).to(device)
    else:
        mu = mu_init.to(device)
    if alpha_init is None:
        alpha = torch.zeros(m).to(device) + 1e-2
    else:
        alpha = alpha_init.to(device)

    for i in range(max_iter):
        
        mu_old = mu.clone()
        alpha_old = alpha.clone()
        eb = alpha * mu

        # CAVI updates
        for j in range(m):
            mu[j] = s2[j] / sigma2 * (xty[j] - torch.sum(xtx[j,:] * eb) + xtx[j,j] * eb[j]) 
            alpha[j] = torch.sigmoid(alpha_const[j] + 0.5 * (mu[j]**2 / s2[j]))
            eb[j] = alpha[j] * mu[j]

        # check convergence
        mu_delta = torch.max(torch.abs(mu - mu_old))
        alpha_delta = torch.max(torch.abs(alpha - alpha_old))

        if mu_delta < delta and alpha_delta < delta:
            print(f"Converged at iteration {i+1}")
            break

    return mu, s2, alpha

def finemap_inf_cavi(z, R, N, tau2, sigma20, pi,  mu1_init=None, mu0_init=None, alpha_init=None, max_iter=100, delta=1e-4, sigma2=1, device="cuda"):
    # compute sufficient statistics
    xty = math.sqrt(N) * z
    xtx = N * R

    # precompute
    m = len(xty)
    s21 = 1.0 / (1.0 / (tau2 + sigma20) + torch.diag(xtx) / sigma2)
    s20 = 1.0 / (1.0 / sigma20 + torch.diag(xtx) / sigma2)
    prior_odds = torch.log(pi) - torch.log(1-pi)

    # initialization
    mu1 = mu1_init
    mu0 = mu0_init
    alpha = alpha_init
    if mu1_init is None:
        mu1 = torch.zeros(m).to(device)
    if mu0_init is None:
        mu0 = torch.zeros(m).to(device)
    if alpha_init is None:
        alpha = torch.zeros(m).to(device) + 1e-2

    for i in range(max_iter):
        
        mu1_old = mu1.clone()
        mu0_old = mu0.clone()
        alpha_old = alpha.clone()
        eb = alpha * mu1 + (1-alpha) * mu0

        # CAVI updates
        for j in range(m):
            resid = xty[j] - torch.sum(xtx[j,:] * eb) 
            resid_no_j = resid + xtx[j,j] * eb[j]
            mu1[j] = s21[j] / sigma2 * resid_no_j
            mu0[j] = s20[j] / sigma2 * resid_no_j

            # Compute alpha_j
            kl0 = 0.5 * ((mu0[j]**2 + s20[j]) / sigma20[j] - torch.log(s20[j]) + torch.log(sigma20[j]) - 1 )
            kl1 = 0.5 * ((mu1[j]**2 + s21[j]) / (tau2[j] + sigma20[j]) - torch.log(s21[j]) + torch.log(tau2[j] + sigma20[j]) - 1)
            kl_gain = kl0 - kl1 
            lkh_gain = (mu1[j] - mu0[j]) / sigma2 * resid - xtx[j,j] / (2 * sigma2) * ((s21[j] - s20[j]) + (1-2*alpha[j]) * (mu1[j] - mu0[j])**2)
            logit_alpha = prior_odds[j] + kl_gain + lkh_gain
            alpha[j] = torch.sigmoid(logit_alpha)
            
            eb[j] = alpha[j] * mu1[j] + (1-alpha[j]) * mu0[j]

        # check convergence
        mu1_delta = torch.max(torch.abs(mu1 - mu1_old))
        mu0_delta = torch.max(torch.abs(mu0 - mu0_old))
        alpha_delta = torch.max(torch.abs(alpha - alpha_old))

        if mu1_delta < delta and mu0_delta < delta and alpha_delta < delta:
            print(f"Converged at iteration {i+1}")
            break

    return mu1, mu0, s21, s20, alpha