# generate data given a biophysical model, a sequencing noise model, and a method of quadrature
# all are fixed for now but can be changed later

# generate training data for bursty model with bernoulli noise

# add path to monod

# +
# import argparse
# parser = argparse.ArgumentParser()

# +
# parser.add_argument('--npdf', type=int)
# args = parser.parse_args()
# npdf = args.npdf
# -

import sys, importlib_resources
package_resources = importlib_resources.files("cme_toolbox")

sys.path.insert(0, package_resources)
sys.path.insert(0, package_resources)


# math
import numpy as np
import random
import scipy.stats as stats


# saving data
import pickle

import monod
import cme_toolbox



# get moments
def get_moments(b,beta,gamma):
    ''' Returns mean, variance, standard deviation and max_n, max_m (mean_z + 20*std_z, where z is species) for
    nascent and mature molecules given parameters included in p.
    
    -------
    
    parameter p:
        b, beta, gamma
    '''

    
    mu_n = b/beta
    mu_m = b/gamma
    
    var_n = (mu_n)*(b + 1)
    var_m = (mu_m)*(b*beta/(beta+gamma) + 1)

    std_n,std_m = np.sqrt(var_n),np.sqrt(var_m)

    max_n,max_m = np.ceil(mu_n + 20*std_n),np.ceil(mu_m + 20*std_m)
    
    # set max to be either 30 or whatever the max returns -- no grids with less than 30 on either axis
    max_n, max_m = int(np.clip(max_n,30,np.inf)),int(np.clip(max_m,30,np.inf))
    
    max_n_store, max_m_store = np.ceil(mu_n + 3*std_n),np.ceil(mu_m + 3*std_m)
    max_n_store, max_m_store = int(np.clip(max_n_store,30,np.inf)),int(np.clip(max_m_store,30,np.inf))

    COV = b**2/(beta + gamma)
    
    return mu_n,mu_m,var_n,var_m,std_n,std_m,max_n,max_m,COV,max_n_store

def get_NORM(npdf,quantiles='cheb'):
    '''' Returns quantiles based on the number of kernel functions npdf. 
    Chebyshev or linear, with chebyshev as default.
    '''
    if quantiles == 'lin':
        q = np.linspace(0,1,npdf+2)[1:-1]
        norm = stats.norm.ppf(q)
        return norm
    if quantiles == 'cheb':
        n = np.arange(npdf)
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2)
        norm = stats.norm.ppf(q)
        return norm

def get_conditional_moments(MU, VAR, COV, n):
    ''' Get moments of conditional distributions (lognormal moment matching) given overall distribution
    mearn, variance, standard deviation, covariance over a range of nascent values.
    '''
    logvar = np.log((VAR/MU**2)+1)
    logstd = np.sqrt(logvar)
    logmean = np.log(MU**2/np.sqrt(VAR+MU**2))

    logcov = np.log(COV * np.exp(-(logmean.sum()+logvar.sum()/2)) +1 ) 
    logcorr = logcov/np.sqrt(logvar.prod())

    logmean_cond = logmean[1] + logcorr * logstd[1]/logstd[0] * (np.log(n+1) - logmean[0])
    logstd_cond = logstd[1] * np.sqrt(1-logcorr**2)   
   
    return(logmean_cond,logstd_cond)

def get_quantile_moments(logmean_cond, logstd_cond, NORM):
    ''' Returns the conditional quantile moments mu_1 to mu_npdf and std_1 to std_npdf for conditional m distributions.
    Moment matches to lognormal, returns quantiles.
    '''
    
    mus = np.exp(logmean_cond + logstd_cond*NORM)
    stds = np.zeros(len(NORM)) 
    stds[:-1] = np.diff(mus)
    stds[-1] = np.sqrt(mus[-1])
    if npdf == 1:
        mus = logmean_cond
        stds = logstd_cond
    
    return mus, stds



# generate parameter vectors
def generate_param_pss_pairs(N, npdf, b_bounds= [0.1,300], beta_bounds = [0.05,50], 
                        gamma_bounds = [0.05,50], params = [0]):
    '''Generates N parameter vectors randomly spaced in logspace between bounds.'''
                    
    # store param, Pss pairs
    p_pss_list = []
    
    if len(params) == 1:
        # generate parameters                
        logbnd = np.log10([b_bounds,beta_bounds,gamma_bounds])
        dbnd = logbnd[:,1]-logbnd[:,0]
        lbnd = logbnd[:,0]
        len_params = N
    else:
        len_params = N

    i = 0
    while ((i<N) and (i<len_params)):
        
        if len(params) == 1:
            log_b,log_beta,log_gamma = np.random.rand(3)*dbnd+lbnd
        else:
            log_b,log_beta,log_gamma = params[i,0],params[i,1],params[i,2]
            if log_beta == log_gamma:
                log_beta += 0.6

        mu_n,mu_m,var_n,var_m,std_n,std_m,max_n,max_m,COV,max_n_store = get_moments(10**log_b,10**log_beta,10**log_gamma)
    
        lim_m = np.max([max_m,300])
        min_m = np.min([max_m,300])
        min_n = np.min([max_n_store,300])
        
        if (max_m < 6000) and (max_n < 6000):
            pss = model.eval_model_pss(np.array([log_b,log_beta,log_gamma]),
                                       np.array([max_n,lim_m],dtype=int))
            print('Shape:',pss.shape)
            
            for n in range(min_n):
                
                pss_ = pss[n,:]
                
                if max(pss_ > 1e-10):
                    param_ = np.ones(4+2*npdf)

                    # get moment matched conditional means and stds
                    logmean_cond, logstd_cond = get_conditional_moments(np.array([mu_n,mu_m]), np.array([var_n,var_m]), COV, n)
                
                    # get the conditional grid parameters
                    mus, stds = get_quantile_moments(logmean_cond,logstd_cond,NORM)
                
                    # store
                    param_[:4] = np.array([log_b,log_beta,log_gamma,n])
                    param_[4:4+npdf] = mus
                    param_[4+npdf:] = stds
                
                    p_pss_list.append([param_,np.float32(pss[n,:300])])
                else:
                    pass
                    
            i+=1
            
    return(p_pss_list)


# define CME model
model = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='quad_vec')


# save_list_train_1 = [f'64_params_{i}_train' for i in range(4)]
# save_list_valid_1 = [f'64_params_{i}_valid' for i in range(2)]
# save_list_test_1 = [f'64_params_{i}_test' for i in range(2)]

save_list_train = [f'256_params_{i}_train' for i in range(15)]
save_list_valid = [f'256_params_{i}_valid' for i in range(2)]
save_list_test = [f'256_params_{i}_test' for i in range(2)]

save_list = save_list_train + save_list_valid + save_list_test

# save_list = ['256_params_4_test']


quantile = 'cheb'
NORM = get_NORM(npdf,quantile)  

for i,s in enumerate(save_list):
    print(f'Starting {s}!')
    
    N = int(s.split('_')[0])
    p_pss_list = generate_param_pss_pairs(N,npdf)
    
    with open(f'../data/{npdf}_npdf_{quantile}/{s}',"wb") as file:
        pickle.dump(p_pss_list,file)
    print(f'Done with {s}!!');
