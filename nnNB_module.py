import numpy as np
import torch
torch.set_default_dtype(torch.float32)

import pickle

import os, sys
sys.path.append('../train_1NB')

import training_nnNB as train



# load in meta

f = '../models/nnNB_models/3hl_256hu_10bs_varmax'
file = open(f + '_meta','rb')
meta = pickle.load(file)
file.close()


# load in model
model1 = train.MLP_1NB_varmax(input_size = 6, num_hidden_units = 256,
                      num_hidden_layers = 3,
                      output_size = 2, 
                      activate='relu',
                      final_activation = 'sigmoid',
                     )

model1.load_state_dict(torch.load(f+'_MODEL'))
model1.eval()


eps = 1e-8

def get_conditional_mean_var(b,beta,gamma,n):
    '''Get moments of conditional distributions (lognormal moment matching) given overall distribution
    mearn, variance, standard deviation, covariance over a range of nascent values.'''
    
    r1,r2 = 1/beta, 1/gamma
    mu1,mu2 = b*r1,b*r2
    var1,var2 = mu1*(1+b), mu2*(1+b*beta/(beta+gamma))
    std1,std2 = np.sqrt(var1),np.sqrt(var2)

    var1 = mu1 * (1+b)
    var2 = mu2 * (1+b*beta/(beta+gamma))
    cov = b**2/(beta+gamma)
    
    logvar1 = np.log((var1/mu1**2)+1)
    logvar2 = np.log((var2/mu2**2)+1)

    logstd1 = np.sqrt(logvar1)
    logstd2 = np.sqrt(logvar2)

    logmean1 = np.log(mu1**2/np.sqrt(var1+mu1**2))
    logmean2 = np.log(mu2**2/np.sqrt(var2+mu2**2))

    logcov = np.log(cov * np.exp(-(logmean1 + logmean2 + (logvar1 + logvar2)/2)) +1 )
    logcorr = logcov/np.sqrt(logvar1 * logvar2)

    logmean_cond = logmean2 + logcorr * logstd2/logstd1 * (np.log(n+1) - logmean1)
    logvar_cond = logvar2 * (1-logcorr**2)  

    mean_cond = np.exp(logmean_cond + logvar_cond/2)
    var_cond = np.exp(2*logmean_cond + logvar_cond) * (np.exp(logvar_cond) - 1)

    return(mean_cond,var_cond)

def log_P(m,mean_cond,var_cond,beta):   
    ''' Returns the LOG of the negative binomial probability given mean and variance at point (or points) m.
    '''
    
    r_cond = mean_cond**2/(var_cond-mean_cond)
    r_cond = r_cond.reshape(-1,1)
    p_cond = mean_cond/var_cond
    p_cond = p_cond.reshape(-1,1)
    r = 1/beta
    r = r.reshape(-1,1)
    mean_cond = mean_cond.reshape(-1,1)
    var_cond = var_cond.reshape(-1,1)

    filt = torch.logical_and(torch.logical_and(r>0,p_cond>0), p_cond<1)
    filt = filt.flatten()
    #compute the Poisson mean
    y_ = m * torch.log(mean_cond+eps) - mean_cond - torch.lgamma(m+1) 

    y_[filt] += torch.lgamma(m[filt]+r_cond[filt]) - torch.lgamma(r_cond[filt]) \
                + r_cond[filt] * torch.log(r_cond[filt]/(r_cond[filt]+mean_cond[filt])+eps) \
                - m[filt] * torch.log(r_cond[filt]+mean_cond[filt]+eps) + mean_cond[filt]
    P =  y_
   
    if torch.any(~torch.isfinite(y_)): 
        print(y_)
        raise ValueError('bad y_')
   
    return P

def get_ypred_log(vecs,m,s_mean,s_var,use_old = False):
    
    if use_old == True:
        print('using old')

    b =  10**vecs[:,0]
    beta =  10**vecs[:,1]
    gamma =  10**vecs[:,2]
    n = vecs[:,3]
    mean_cond = vecs[:,4]
    var_cond = vecs[:,5]
    

    mean_cond_scaled = s_mean*mean_cond 
    var_cond_scaled = s_var*var_cond 
    
    if use_old == True:
        log_y = log_P(m,mean_cond,var_cond,beta)
        'USING MMNB'
    else: 
        log_y = log_P(m,mean_cond_scaled,var_cond_scaled,beta)
    
    return(log_y)



def nnNB_prob(p,n,m,model,use_old=False):
    ''' Calculates probability for bursty model using input model.
    '''
    
    log_b =  p[0] * np.ones(len(n))
    b = 10**log_b
    log_beta =  p[1] * np.ones(len(n))
    beta = 10**log_beta
    log_gamma =  p[2] * np.ones(len(n))
    gamma = 10**log_gamma
    
    
    # right now, assume b,beta,gamma,n are of shape (X,1) -- can change later
    mean_cond,var_cond = get_conditional_mean_var(b,beta,gamma,n)
    
    log_b,log_beta,log_gamma = torch.tensor(log_b,dtype=torch.float32), \
                        torch.tensor(log_beta,dtype=torch.float32), \
                            torch.tensor(log_gamma,dtype=torch.float32) 
    n,mean_cond,var_cond = torch.tensor(n,dtype=torch.float32), \
                torch.tensor(mean_cond,dtype=torch.float32),  \
                    torch.tensor(var_cond,dtype=torch.float32)
    m = torch.tensor(m,dtype=torch.float32)
    vecs = torch.column_stack((log_b,log_beta,log_gamma,n,mean_cond,var_cond))
    
    # feed through the model
    s_mean, s_var = model(vecs)
    
    m = m.repeat(1,len(n)).reshape((len(n)),-1)

    # calculate the probability -- will be an array [n,m]
    prob_cond_log = get_ypred_log(vecs,m,s_mean,s_var,use_old=use_old)
    
    # negative binomial of n
    r = 1/torch.tensor(beta)
    MU = torch.tensor(b)*r
    
    prefactor_log = torch.lgamma(n+r) - torch.lgamma(n+1) - torch.lgamma(r) \
                + r * torch.log(r/(r+MU)) + n * torch.log(MU/(r+MU))
    

    
    
    prob_cond,prefactor = torch.exp(prob_cond_log),torch.exp(prefactor_log)

    
    P = prob_cond*prefactor.reshape(-1,1)
    
    P = P.detach().numpy()
    
    return(P)