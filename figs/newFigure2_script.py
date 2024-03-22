#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys,os

sys.path.append('../psKWR/')
sys.path.append('../monod/src/')
sys.path.append('../monod/src/monod/')

# numbers and dataloaders
import numpy as np
import pickle

# monod
from monod import cme_toolbox 
from nn_toolbox import get_moments, get_conditional_moments, get_quantile_moments, get_NORM

sys.path.append('../')

import ypred_module as ypm
import train_conditional as train
import direct_module as direct

import time


# In[18]:


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


# In[19]:


# Figure 2a: Time vs. Hellinger distance
data_set = train.load_data(1,'../data/KWR_data/','256_test_full')


# In[20]:


# set up monod models 
fitmodel_qv = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='quad_vec')
fitmodel_fq = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='fixed_quad')
fitmodel_KWR = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='nn_10')
fitmodel_psKWR = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='nn_microstate')


# In[21]:


# store 
timing = {'state_spaces' : np.ones(len(data_set)),
          'params' : [],
          'QV20' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'QV10' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'QV4' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'QV1' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'FQ' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'KWR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'psKWR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'MMNB' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'RW' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'DR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))}
         }

hellinger = { 'state_spaces' : np.ones(len(data_set)),
          'QV10' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'QV4' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'QV1' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'FQ' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'KWR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'psKWR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'MMNB' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'RW' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'DR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))}
         }

kld = { 'state_spaces' : np.ones(len(data_set)),
          'QV10' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'QV4' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'QV1' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'FQ' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'KWR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'psKWR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'MMNB' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'RW' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))},
          'DR' : {'norm' : np.ones(len(data_set)), 'unnorm' : np.ones(len(data_set))}
         }
    
    


# In[ ]:

for i in range(256):
    print(i)
    p_ = data_set[i][0]
    mu_n,mu_m,var_n,var_m,std_n,std_m,COV = get_moments(10**p_[0],10**p_[1],10**p_[2])
    lim_ = [int(np.max([np.ceil(mu_n+1*std_n),10])),int(np.max([np.ceil(mu_m+1*std_m),10]))]
    ss_ = lim_[0]*lim_[1]
    
    # QV20
    lim_large = [int(np.max([np.ceil(mu_n+20*std_n),10])),int(np.max([np.ceil(mu_m+20*std_m),10]))]
    print(ss_)
    timing['state_spaces'][i] = ss_
    timing['params'].append(p_)
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
    kld_ = get_hellinger(qv20[:lim_[0],:lim_[1]],fq[:lim_[0],:lim_[1]])
    kld['FQ']['unnorm'][i]= kld_
    kld['FQ']['norm'][i]=kld_/ss_
    
    # KWR
    if i == 0:
        # burn in
        kwr = fitmodel_KWR.eval_model_pss(p_,limits=lim_)
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
    
    
    # RW
    nas_range = np.arange(lim_[0])
    mat_range = np.arange(lim_[1])
    t1 = time.time()
    rw = ypm.get_prob(p_,nas_range,mat_range,rand_weights=True)
    t_ = time.time()-t1
    timing['RW']['unnorm'][i]=t_
    timing['RW']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],rw)
    hellinger['RW']['unnorm'][i]= hell_
    hellinger['RW']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],rw)
    kld['RW']['unnorm'][i]= kld_
    kld['RW']['norm'][i]= kld_/ss_
    
    # Direct
    t1 = time.time()
    dr = direct.predict_pmf(p_,lim_[0],lim_[1])
    t_ = time.time()-t1
    timing['DR']['unnorm'][i]=t_
    timing['DR']['norm'][i]=t_/ss_
    hell_ = get_hellinger(qv20[:lim_[0],:lim_[1]],dr)
    hellinger['DR']['unnorm'][i]= hell_
    hellinger['DR']['norm'][i]=hell_/ss_
    kld_ = get_kld(qv20[:lim_[0],:lim_[1]],dr)
    kld['DR']['unnorm'][i]= kld_
    kld['DR']['norm'][i]= kld_/ss_
    


# In[ ]:


# save!!!! 
import pickle 

with open('./new_hellinger_dict_1std', 'wb') as file:
    pickle.dump(hellinger, file)
    
with open('./new_kld_dict_1std', 'wb') as file:
    pickle.dump(kld, file)
    
with open('./new_timing_dict_1std', 'wb') as file:
    pickle.dump(timing, file)



# In[ ]:


# single microstate timing
microstate_timing = {
          'KWR' : np.ones(len(data_set)),
          'psKWR' : np.ones(len(data_set)),
          'MMNB' : np.ones(len(data_set)),
          'RW' : np.ones(len(data_set)),
          'DR' : np.ones(len(data_set)),
         }


# In[ ]:


# new microstate timing vs. full grid for 256 parameters

for i in range(256):
  
    lim_ = [int(1),int(1)]
    
    # KWR
    t1 = time.time()
    kwr = fitmodel_KWR.eval_model_pss(p_,limits=lim_)
    t_ = time.time()-t1
    microstate_timing['KWR'][i]=t_
    
    # psKWR
    t1 = time.time()
    pskwr = fitmodel_psKWR.eval_model_pss(p_,limits=lim_)
    t_ = time.time()-t1
    microstate_timing['psKWR'][i]=t_
    
    # MMNB
    N,M = np.meshgrid(range(lim_[0]),range(lim_[1]),indexing='ij')
    t1 = time.time()
    mmnb = ypm.approximate_conditional_tensorval(p_,N,M).detach().numpy()
    t_ = time.time()-t1
    microstate_timing['MMNB'][i]=t_
    
    # RW
    nas_range = np.arange(lim_[0])
    mat_range = np.arange(lim_[1])
    t1 = time.time()
    rw = ypm.get_prob(p_,nas_range,mat_range,rand_weights=True)
    t_ = time.time()-t1
    microstate_timing['RW'][i]=t_

    
    # Direct
    t1 = time.time()
    dr = direct.predict_pmf(p_,lim_[0],lim_[1])
    t_ = time.time()-t1
    microstate_timing['DR'][i]=t_


# In[ ]:


with open('./new_microstate_timing_dict', 'wb') as file:
    pickle.dump(microstate_timing, file)

