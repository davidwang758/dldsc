import torch
import math
import torch.distributions as distributions

# Alt BF calc
# bf = torch.sqrt(varbhat / (tau2 + varbhat)) * torch.exp(bhat**2 / (2 * varbhat) * tau2 / (tau2 + varbhat))
# alpha = pi * bf / (torch.sum(pi * bf))

def SER_rss_updates(xtx, xtr, tau2, pi, sigma2):
    xtx_j = torch.diag(xtx)
    bhat = xtr / xtx_j
    varbhat = sigma2 / xtx_j
    s2 = 1 / (1 / tau2 + 1 / varbhat)
    mu =  s2 * bhat / varbhat

    lbf = distributions.Normal(loc=0., scale=torch.sqrt(tau2 + varbhat)).log_prob(bhat) - \
          distributions.Normal(loc=0., scale=torch.sqrt(varbhat)).log_prob(bhat)

    weighted_lbf = torch.log(pi) + lbf
    alpha = torch.exp(weighted_lbf - torch.logsumexp(weighted_lbf, dim=0))

    return mu, s2, alpha

def susie_ibss(z, R, N, tau2, pi, L=10, mu_init=None, alpha_init=None, max_iter=100, delta=1e-4, sigma2=1, device="cuda"):
    # Convert summary stats into sufficient stats
    xty = math.sqrt(N) * z
    xtx = R * N

    # Initialize
    
    if mu_init is None:
        mu = torch.zeros((L, len(z))).to(device)
    else:
        mu = mu_init.to(device)
    if alpha_init is None:
        alpha = torch.zeros((L, len(z))).to(device)
    else: 
        alpha = alpha_init.to(device)
    s2 = torch.zeros((L, len(z))).to(device)

    Eb = mu * alpha
    xtr = xty - torch.sum(Eb, axis=0) @ xtx
        
    for i in range(max_iter):
        mu_old = mu.clone()
        s2_old = s2.clone()
        alpha_old = alpha.clone()
        
        for l in range(L):
            # Remove effect l from residuals and update posteriors with SER treating residuals as xty
            xtr_l = xtr + Eb[l,:] @ xtx         
            mu[l,:], s2[l,:], alpha[l,:] = SER_rss_updates(xtx, xtr_l, tau2[l], pi, sigma2)
            Eb[l,:] = alpha[l,:] * mu[l,:]
            xtr = xtr_l - Eb[l,:] @ xtx 

        mu_delta = torch.max(torch.abs(mu - mu_old))
        s2_delta = torch.max(torch.abs(s2 - s2_old))
        alpha_delta = torch.max(torch.abs(alpha - alpha_old))

        if mu_delta < delta and s2_delta < delta and alpha_delta < delta:
            print(f"Converged at iteration {i+1}")
            break

    # Return posterior second momemnt instead of posterior variance
    return(mu, (mu**2 + s2), alpha)

#bhat = xtr / xtx_j
#varbhat = sigma2 / xtx_j
#s2 = 1 / (1 / tau2 + 1 / varbhat)
#mu =  s2 * bhat / varbhat

#lbf = distributions.Normal(loc=0., scale=torch.sqrt(tau2 + varbhat)).log_prob(bhat) - \
#      distributions.Normal(loc=0., scale=torch.sqrt(varbhat)).log_prob(bhat)
def multitask_SER_rss(xtx, xtr, tau2, sigma2, pi, coloc):
    xtx_j = torch.diagonal(xtx).unsqueeze(1)
    ratio = 1.0 / (1.0 + (tau2 / sigma2) * xtx_j)
    lbf = 0.5 * (torch.log(ratio) + (xtr ** 2) / (sigma2 * xtx_j) * (1.0 - ratio))
    del ratio

    if coloc:
        lbf = torch.log(pi) + torch.sum(lbf, axis=1)
    else:
        lbf = torch.log(pi) + lbf

    alpha = torch.exp(lbf - torch.logsumexp(lbf, dim=0))
    del lbf

    s2 = 1 / (1 / tau2 + xtx_j / sigma2)
    mu = s2 * xtr / sigma2
    return mu, s2, alpha

def multitask_susie_ibss(z, R, N, tau2, pi, L=10, Eb_init=None, max_iter=100, delta=1e-4, sigma2=1, device="cuda", coloc=False):
    m, v = z.shape
    xty = math.sqrt(N) * z
    del z
    xtx = N * R
    del R
    
    mu = torch.zeros((L, m, v), device=device)
    s2 = torch.zeros((L, m, v), device=device)

    if coloc:
        alpha = torch.zeros((L, m), device=device)
        Eb = alpha.unsqueeze(2) * mu
    else:
        alpha = torch.zeros((L, m, v), device=device)
        Eb = alpha * mu
    if Eb_init is not None:
        l_init, v_init, val = Eb_init
        Eb[l_init.to(device),:,v_init.to(device)] = val.to(device=device, dtype=torch.float32)
        
    xtr = xty - torch.einsum('iv, ij -> jv', torch.sum(Eb, axis=0), xtx)

    active = torch.ones(v, dtype=torch.bool, device=device)

    for i in range(max_iter):
        #mu_old = mu[:,:,active].clone()
        #s2_old = s2[:,:,active].clone()
        #alpha_old = alpha[:,:,active].clone()    
           
        current_max = torch.zeros(torch.sum(active), device=device) 

        for l in range(L):
            #xtr_l = xtr + torch.einsum('iv, ij -> jv', Eb[l,:,:], xtx)
            xtr[:,active] = xtr[:,active] + torch.einsum('iv, ij -> jv', Eb[l,:,active], xtx)
            #mu[l,:,:], s2[l,:,:], alpha_update, lbf_update = multitask_SER_rss(xtx, xtr_l, tau2[l], sigma2, pi, coloc)

            mu_old_l = mu[l,:,active].clone()
            s2_old_l = s2[l,:,active].clone()
            alpha_old_l = alpha[l,:,active].clone() 

            if coloc:
                mu[l,:,active], s2[l,:,active], alpha[l,:] = multitask_SER_rss(xtx, xtr[:,active], tau2[l], sigma2, pi[:,active], coloc)
                Eb[l,:,active] = alpha[l,:].unsqueeze(1) * mu[l,:,active]
            else:
                #mu[l,:,:], s2[l,:,:], alpha[l,:,:], lbf[l,:,:] = multitask_SER_rss(xtx, xtr_l, tau2[l], sigma2, pi, coloc)
                #Eb[l,:,:] = alpha[l,:,:] *  mu[l,:,:]
                
                mu[l,:,active], s2[l,:,active], alpha[l,:,active] = multitask_SER_rss(xtx, xtr[:,active], tau2[l], sigma2, pi[:,active], coloc)
                Eb[l,:,active] = alpha[l,:,active] *  mu[l,:,active]

            xtr[:,active] = xtr[:,active] - torch.einsum('iv, ij -> jv', Eb[l,:,active], xtx)

            mu_delta_l = torch.amax(torch.abs(mu[l,:,active] - mu_old_l), dim=0)
            s2_delta_l = torch.amax(torch.abs(s2[l,:,active] - s2_old_l), dim=0)
            alpha_delta_l = torch.amax(torch.abs(alpha[l,:,active] - alpha_old_l), dim=0)
            current_max = torch.maximum(current_max,torch.maximum(mu_delta_l, torch.maximum(s2_delta_l, alpha_delta_l)))

        #active[active.clone()] = (mu_delta >= delta) | (s2_delta >= delta) | (alpha_delta >= delta)
        active[active.clone()] = current_max >= delta 

        if torch.all(active == False):
            #print(f"Converged at iteration {i+1}")
            break

        #if mu_delta < delta and s2_delta < delta and alpha_delta < delta:
        #    print(f"Converged at iteration {i+1}")
        #    break

    return mu, (mu**2 + s2), alpha # You only need the LBF during inference.

def multitask_SER_rss_original(xtx, xtr, tau2, sigma2, pi, coloc):
    # Extract diagonal and reshape for broadcasting
    xtx_j = torch.diagonal(xtx).unsqueeze(1)
    
    # Calculate bhat and varbhat explicitely (Allocates memory)
    bhat = xtr / xtx_j
    varbhat = sigma2 / xtx_j
    
    # Posterior calculations
    s2 = 1 / (1 / tau2 + 1 / varbhat)
    mu = s2 * bhat / varbhat

    # Heavy: Creating Distribution objects and calculating log_prob
    # This allocates multiple intermediate tensors for the graph
    prior_dist = distributions.Normal(loc=0., scale=torch.sqrt(tau2 + varbhat))
    null_dist = distributions.Normal(loc=0., scale=torch.sqrt(varbhat))
    
    lbf = prior_dist.log_prob(bhat) - null_dist.log_prob(bhat)

    if coloc:
        # Sum across traits
        lbf = torch.log(pi) + torch.sum(lbf, dim=1)
    else:
        lbf = torch.log(pi) + lbf
        
    # Softmax to get PIPs
    # Note: This original version often crashed with NaN if lbf was too large
    alpha = torch.exp(lbf - torch.logsumexp(lbf, dim=0))

    return mu, s2, alpha, lbf

def multitask_susie_ibss_old(z, R, N, tau2, pi, L=10, mu_init=None, alpha_init=None, max_iter=100, delta=1e-4, sigma2=1, device="cuda", coloc=False):
    m, v = z.shape
    
    # 1. Transform Data (Creates copies, keeps originals z/R in memory)
    xty = math.sqrt(N) * z
    xtx = N * R
    
    # Initialize Model
    if mu_init is None:
        mu = torch.zeros((L, m, v)).to(device) 
    else:
        mu = mu_init.to(device)
        
    s2 = torch.zeros((L, m, v)).to(device)
    
    # Allocates full lbf tensor (L x M x V) which is huge and unused
    lbf = torch.zeros((L, m, v)).to(device)

    if coloc:
        if alpha_init is None:
            alpha = torch.zeros((L, m)).to(device) # Note: broadcasting mismatch in original logic often occurred here
        else:
            alpha = alpha_init.to(device)
        Eb = alpha.unsqueeze(2) * mu
    else:
        if alpha_init is None:
            alpha = torch.zeros((L, m, v)).to(device)
        else:
            alpha = alpha_init.to(device)
        Eb = alpha * mu

    # Initial Residuals
    xtr = xty - torch.einsum('iv, ij -> jv', torch.sum(Eb, axis=0), xtx)

    # Active set logic (Original: boolean mask, often logic was slightly buggy regarding shapes)
    active = torch.ones(v, dtype=torch.bool).to(device)

    for i in range(max_iter):
        # 2. Huge Memory Spike: Cloning the ENTIRE model for delta check
        mu_old = mu.clone()
        s2_old = s2.clone()
        alpha_old = alpha.clone()
        
        for l in range(L):
            # Remove effect l from residuals
            if coloc:
                Eb[l] = alpha[l].unsqueeze(1) * mu[l]
            else:
                Eb[l] = alpha[l] * mu[l]
                
            xtr = xtr + torch.einsum('iv, ij -> jv', Eb[l], xtx)

            # Run SER (Passing full matrices, no slicing optimization)
            if coloc:
                # Note: Original code often had shape mismatches here for alpha
                mu[l], s2[l], alpha[l], lbf[l] = multitask_SER_rss_original(xtx, xtr, tau2[l], sigma2, pi, coloc)
                Eb[l] = alpha[l].unsqueeze(1) * mu[l]
            else:
                mu[l], s2[l], alpha[l], lbf[l] = multitask_SER_rss_original(xtx, xtr, tau2[l], sigma2, pi, coloc)
                Eb[l] = alpha[l] * mu[l]

            # Add effect l back to residuals
            xtr = xtr - torch.einsum('iv, ij -> jv', Eb[l], xtx)

        # 3. Convergence Check (Calculates delta on full tensors)
        mu_delta = torch.amax(torch.abs(mu - mu_old))
        s2_delta = torch.amax(torch.abs(s2 - s2_old))
        alpha_delta = torch.amax(torch.abs(alpha - alpha_old))

        if mu_delta < delta and s2_delta < delta and alpha_delta < delta:
            print(f"Converged at iteration {i+1}")
            break

    return mu, (mu**2 + s2), alpha