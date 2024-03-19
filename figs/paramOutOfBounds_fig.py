
import numpy as np
import time
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str) # options 'b' 'beta' 'gamma' 'all'
args = parser.parse_args()

name = args.name

import sys,os

sys.path.append('../psKWR/')
sys.path.append('../monod/src/')
sys.path.append('../monod/src/monod/')


# monod
from monod import cme_toolbox 
from nn_toolbox import get_moments, get_conditional_moments, get_quantile_moments, get_NORM

sys.path.append('../')

import ypred_module as ypm
import train_conditional as train



eps = 1e-18

def get_hellinger(p,q):
    p = p.flatten()
    q = q.flatten()
    p_sqrt = np.sqrt(p)
    q_sqrt = np.sqrt(q)
    a = (p_sqrt-q_sqrt)**2
    b = np.sqrt(np.sum(a))
    
    return (1/(np.sqrt(2)))*b
    
def get_kld(p,q):
    p = p/p.sum()
    q = q/q.sum()
    p = p.flatten()
    q = q.flatten()
    kld = np.sum( p*np.log(p/q + eps) )
    return kld
  
# load in parameters
params = np.load(f'../data/paramsOutOfBounds/params_{name}_out.npy')
    
# set up monod models 
fitmodel_qv = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='quad_vec')
fitmodel_fq = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='fixed_quad')
fitmodel_KWR = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='nn_10')
fitmodel_psKWR = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='nn_microstate')

# store 
timing = {'state_spaces' : np.ones(len(params)),
          'QV20' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'QV10' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'QV4' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'QV1' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'FQ' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'KWR' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'psKWR' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'MMNB' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
         }

hellinger = { 'state_spaces' : np.ones(len(params)),
          'QV10' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'QV4' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'QV1' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'FQ' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'KWR' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'psKWR' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'MMNB' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
         }
    
kld = { 'state_spaces' : np.ones(len(params)),
          'QV10' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'QV4' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'QV1' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'FQ' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'KWR' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'psKWR' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
          'MMNB' : {'norm' : np.ones(len(params)), 'unnorm' : np.ones(len(params))},
         }
         
         
for i,p_ in enumerate(params):
    print(i)
    
    mu_n,mu_m,var_n,var_m,std_n,std_m,COV = get_moments(10**p_[0],10**p_[1],10**p_[2])
    lim_ = [int(np.max([np.ceil(mu_n+1*std_n),10])),int(np.max([np.ceil(mu_m+1*std_m),10]))]
    ss_ = lim_[0]*lim_[1]
    
    # QV20
    lim_large = [int(np.max([np.ceil(mu_n+20*std_n),10])),int(np.max([np.ceil(mu_m+20*std_m),10]))]
    print(ss_)
    timing['state_spaces'][i] = ss_
    hellinger['state_spaces'][i] = ss_
    t1 = time.time()
    qv20 = fitmodel_qv.eval_model_pss(p_,limits=lim_large)
    t_ = time.time()-t1
    timing['QV20']['unnorm'][i]=t_
    timing['QV20']['norm'][i]=t_/ss_
    
    # QV10
    lim_large = [int(np.max([np.ceil(mu_n+10*std_n),10])),int(np.max([np.ceil(mu_m+10*std_m),10]))]
    t1 = time.time()
    qv10 = fitmodel_qv.eval_model_pss(p_,limits=lim_large)
    t_ = time.time()-t1
    timing['QV10']['unnorm'][i]=t_
    timing['QV10']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],qv10[:lim_[0],:lim_[1]])
    hellinger['QV10']['unnorm'][i]= hell_
    hellinger['QV10']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],qv10[:lim_[0],:lim_[1]])
    kld['QV10']['unnorm'][i]= kld_
    kld['QV10']['norm'][i]=kld_/ss_
    
    # QV4
    lim_large = [int(np.max([np.ceil(mu_n+4*std_n),10])),int(np.max([np.ceil(mu_m+4*std_m),10]))]
    t1 = time.time()
    qv4 = fitmodel_qv.eval_model_pss(p_,limits=lim_large)
    t_ = time.time()-t1
    timing['QV4']['unnorm'][i]=t_
    timing['QV4']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],qv4[:lim_[0],:lim_[1]])
    hellinger['QV4']['unnorm'][i]= hell_
    hellinger['QV4']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],qv4[:lim_[0],:lim_[1]])
    kld['QV4']['unnorm'][i]= kld_
    kld['QV4']['norm'][i]=kld_/ss_
    
    # QV1
    t1 = time.time()
    qv1 = fitmodel_qv.eval_model_pss(p_,limits=lim_)
    t_ = time.time()-t1
    timing['QV1']['unnorm'][i]=t_
    timing['QV1']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],qv1)
    hellinger['QV1']['unnorm'][i]= hell_
    hellinger['QV1']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],qv1[:lim_[0],:lim_[1]])
    kld['QV1']['unnorm'][i]= kld_
    kld['QV1']['norm'][i]=kld_/ss_
    
    # FQ
    t1 = time.time()
    fq = fitmodel_fq.eval_model_pss(p_,limits=lim_)
    t_ = time.time()-t1
    timing['FQ']['unnorm'][i]=t_
    timing['FQ']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],fq)
    hellinger['FQ']['unnorm'][i]= hell_
    hellinger['FQ']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],fq[:lim_[0],:lim_[1]])
    kld['FQ']['unnorm'][i]= kld_
    kld['FQ']['norm'][i]=kld_/ss_
    
    # KWR
    t1 = time.time()
    kwr = fitmodel_KWR.eval_model_pss(p_,limits=lim_)
    t_ = time.time()-t1
    timing['KWR']['unnorm'][i]=t_
    timing['KWR']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],kwr)
    hellinger['KWR']['unnorm'][i]= hell_
    hellinger['KWR']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],kwr[:lim_[0],:lim_[1]])
    kld['KWR']['unnorm'][i]= kld_
    kld['KWR']['norm'][i]=kld_/ss_
    
    # psKWR
    t1 = time.time()
    pskwr = fitmodel_psKWR.eval_model_pss(p_,limits=lim_)
    t_ = time.time()-t1
    timing['psKWR']['unnorm'][i]=t_
    timing['psKWR']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],pskwr)
    hellinger['psKWR']['unnorm'][i]= hell_
    hellinger['psKWR']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],pskwr[:lim_[0],:lim_[1]])
    kld['psKWR']['unnorm'][i]= kld_
    kld['psKWR']['norm'][i]=kld_/ss_
    
    # MMNB
    N,M = np.meshgrid(range(lim_[0]),range(lim_[1]),indexing='ij')
    t1 = time.time()
    mmnb = ypm.approximate_conditional_tensorval(p_,N,M).detach().numpy()
    t_ = time.time()-t1
    timing['MMNB']['unnorm'][i]=t_
    timing['MMNB']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],mmnb)
    hellinger['MMNB']['unnorm'][i]= hell_
    hellinger['MMNB']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],mmnb[:lim_[0],:lim_[1]])
    kld['MMNB']['unnorm'][i]= kld_
    kld['MMNB']['norm'][i]=kld_/ss_
    
# SAVE
# save!!!! 
import pickle 

with open(f'./timing_dict_{name}', 'wb') as file:
    pickle.dump(timing, file)
    
with open(f'./hellinger_dict_{name}', 'wb') as file:
    pickle.dump(hellinger, file)
    
with open(f'./kld_dict_{name}', 'wb') as file:
    pickle.dump(kld, file)
