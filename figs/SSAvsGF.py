#!/usr/bin/env python
# coding: utf-8

# # Comparing SSA distribution realizations with GF solutions

# In[43]:


import scipy.stats as stats
import numpy as np
from numpy import matlib as mb

# plotting 
import matplotlib.pyplot as plt


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


# Gennady Gorin's code from: https://github.com/pachterlab/GP_2021_2/blob/main/dag_cme_burst.py

# In[49]:


def gillvec(k,t_matrix,S,nCells):
	k = mb.repmat(k,nCells,1)
	n_species = S.shape[1]

	num_t_pts = t_matrix.shape[1]
	X_mesh = np.empty((nCells,num_t_pts,n_species),dtype=float) #change to float if storing floats!!!!!!! 
	X_mesh[:] = np.nan

	t = np.zeros(nCells,dtype=float)
	tindex = np.zeros(nCells,dtype=int)

	#initialize state: gene,integer unspliced, integer spliced 
	X = np.zeros((nCells,n_species))

	#initialize list of cells that are being simulated
	simindices = np.arange(nCells)
	activecells = np.ones(nCells,dtype=bool)

	while any(activecells):
		mu = np.zeros(nCells,dtype=int);
		n_active_cells = np.sum(activecells)
		
		(t_upd,mu_upd) = rxn_calculator( 			X[activecells,:], 			t[activecells], 			k[activecells,:], 			S, 			n_active_cells)

		t[activecells] = t_upd
		mu[activecells] = mu_upd
		
		tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
		update = np.zeros(nCells,dtype=bool)
		update[activecells] = t[activecells]>tvec_time
		
		while any(update):
			tobeupdated = np.where(update)[0]
			for i in range(len(tobeupdated)):
				X_mesh[simindices[tobeupdated[i]],tindex[tobeupdated[i]],:] = 					X[tobeupdated[i],:]
			
			tindex = tindex + update;
			ended_in_update = tindex[update]>=num_t_pts;

			if any(ended_in_update):
				ended = tobeupdated[ended_in_update];
				
				activecells[ended] = False;
				mu[ended] = 0;

				if ~any(activecells):
					break			
			
			tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
			update = np.zeros(nCells,dtype=bool)
			update[activecells] = t[activecells]>tvec_time
		
		z_ = np.where(activecells)[0]
		not_burst = mu[z_] > 1
		burst = mu[z_] == 1
		if any(not_burst):
			X[z_[not_burst]] += S[mu[z_[not_burst]]-1]
		if any(burst):				 
			bs = np.random.geometric(1/(1+S[0][0]),size=(sum(burst),1))-1
			X[z_[burst]] += mb.hstack((bs,np.zeros((sum(burst),n_species-1),dtype=int)))
	return X_mesh

def rxn_calculator(X,t,k,S,nCells):
	nRxn = S.shape[0]

	kinit = k[:,0]

	a = np.zeros((nCells,nRxn),dtype=float)
	a[:,0] = kinit
	for i in range(1,nRxn):
		ind = np.where(S[i,:]==-1)[0][0]
		a[:,i] = X[:,ind]*k[:,i]

	a0 = np.sum(a,1)
	t += np.log(1./np.random.rand(nCells)) / a0
	r2ao = a0 * np.random.rand(nCells)
	mu = np.sum(mb.repmat(r2ao,nRxn+1,1).T >= np.cumsum(mb.hstack((np.zeros((nCells,1)),a)),1) ,1)
	return (t,mu)


def get_ssa_density(X):
    
    # Find unique pairs and count their occurrences
    unique_pairs_N, counts_N = np.unique(list(zip(X[:,0], X[:,1])), axis=0, return_counts=True)

    # Define the range for the unique pairs
    max_val = max(np.max(X[:,0]), np.max(X[:,1])) + 1
    # unique_pairs_range = np.arange(max_val)

    # Initialize count matrix
    count_matrix = np.zeros( (int(np.max(X[:,0])+1), int(np.max(X[:,1])+1) ) )

    # Fill count matrix with counts
    for pair, count in zip(unique_pairs_N, counts_N):
        count_matrix[int(pair[0]), int(pair[1])] = count
        
    return(count_matrix/len(X))


# # SSA: Gillespie's Algorithm

# In[27]:


# b = 2
# beta = 0.1
# gamma = 3.0
# T = (1/4 + 1/10 + 1)*10
# print(b/beta,b/gamma)


# In[35]:


# S = np.zeros((3,2))

# S[0,0] = b # transcription
# S[1,0] = -1 # splicing, removal of one nascent
# S[1,1] = 1 # splicing, addition of one mature
# S[2,0] = 0 # degradation, no change to nascent
# S[2,1] = -1 # degradation, removal of one mature

# k = [1, beta, gamma]

# nCells = 100000
# T = 10
# measure_time = T/min(k)
# print(measure_time)
# tvec = np.linspace(0,measure_time,2,endpoint=True)
# t_matrix = mb.repmat(tvec,nCells,1)
# X=gillvec(k,t_matrix,S,nCells)
# X = X[:,-1,:]


# # CME generating function solution

# In[36]:


fitmodel_qv = cme_toolbox.CMEModel(bio_model='Bursty',seq_model='None',quad_method='quad_vec')


# In[38]:



# fig,ax = plt.subplots(1,2,figsize = (8,4))
# ax[0].imshow(pmf)
# ax[0].invert_yaxis()


# # Create heatmap
# ax[1].imshow(count_matrix/len(X), cmap='viridis', origin='lower')
# # .colorbar(label='Density of Observed Counts')
# ax[0].set_xlabel('# mature RNA')
# ax[0].set_ylabel('# nascent RNA')
# # plt.xticks(unique_pairs_range)
# # plt.yticks(unique_pairs_range)
# plt.show()


# # Simulate trajectories for 100 parameters

# In[50]:


data_set = train.load_data(1,'../data/KWR_data/','256_test_full')


# In[52]:


ssa_dict = {}

for i in range(100):
    
    print(i)
    key = f'param_{i}'
    ssa_dict[key] = {}
    
    p_ = data_set[i][0]
    b,beta,gamma = 10**p_
    
    S = np.zeros((3,2))

    S[0,0] = b # transcription
    S[1,0] = -1 # splicing, removal of one nascent
    S[1,1] = 1 # splicing, addition of one mature
    S[2,0] = 0 # degradation, no change to nascent
    S[2,1] = -1 # degradation, removal of one mature

    k = [1, beta, gamma]

    nCells = 100000
    T = 10
    measure_time = T/min(k)
    print(measure_time)
    tvec = np.linspace(0,measure_time,2,endpoint=True)
    t_matrix = mb.repmat(tvec,nCells,1)
    X=gillvec(k,t_matrix,S,nCells)
    X = X[:,-1,:]
    
    
    ssa_dict[key]['X'] = X
    ssa_dict[key]['Density_10'] = get_ssa_density(X[:10,:])
    ssa_dict[key]['Density_100'] = get_ssa_density(X[:100,:])
    ssa_dict[key]['Density_1000'] = get_ssa_density(X[:1000,:])
    ssa_dict[key]['Density_10000'] = get_ssa_density(X[:10000,:])
    ssa_dict[key]['Density_100000'] = get_ssa_density(X[:,:])
    
    # GF solution
    mu_n,mu_m,var_n,var_m,std_n,std_m,COV = get_moments(b,beta,gamma)
    lim = [int(np.max([np.ceil(mu_n+20*std_n),int(np.max(X[:,0])+1)])),
       int(np.max([np.ceil(mu_m+20*std_m),int(np.max(X[:,1])+1)]))] 
    lim_plot = [int(np.max(X[:,0])+1), int(np.max(X[:,1])+1)]
    pmf = fitmodel_qv.eval_model_pss(p_,limits=lim)[:lim_plot[0],:lim_plot[1]]
    ssa_dict[key]['PMF_QV20'] = pmf
    
# save!!

with open('../results/ssa_density_dict', 'wb') as file:
    pickle.dump(ssa_dict, file)

    
new_results_dict = {}


for i in range(100):
    key = f'param_{i}'
    new_results_dict[key] = {}
    rd_ = ssa_dict[key]
    lim_ = rd_[f'Density_10'].shape
    pmf_ = rd_[f'PMF_QV20'][:lim_[0],:lim_[1]]
    new_results_dict[key]['PMF_QV20'] = pmf_
    
    for num_traj in [10,100,1000,10000,100000]:
        ssa_dens_ = rd_[f'Density_{num_traj}'][:lim_[0],:lim_[1]]
        # only save up to limits
        new_results_dict[key][f'Density_{num_traj}'] = ssa_dens_

with open('./ssa_density_dict_small', 'wb') as file:
    pickle.dump(new_results_dict, file)
    
# ----------
# 
# My old SSA code.

# In[24]:


# def simulate_one_trajectory(b,beta,gamma,stopping_time=None):
    
#     # initial state
#     N = 0
#     M = 0
#     t = 0
#     # prob of each rxn occurring
    
#     while t < stopping_time:
#         p_txn = 1 
#         p_splicing = N*beta
#         p_deg = M*gamma
#         p = np.array([p_txn/(p_txn+p_splicing+p_deg),p_splicing/(p_txn+p_splicing+p_deg),p_deg/(p_txn+p_splicing+p_deg)])
#         rxn_ = np.random.choice([1,2,3],p = p)
        
#         if rxn_ == 1:
#             N +=  stats.geom.rvs(p=1/b, size=1)[0]
# #         if (rxn_ == 2) and (N > 0):
# #             N-=1
# #             M+=1
# #         if (rxn_ == 3) and (M > 0):
# #             M-=1
#         if (rxn_ == 2):
#             N-=1
#             M+=1
#         if (rxn_ == 3):
#             M-=1
# #         t+= stats.expon.rvs(scale = 1/ (p_txn+p_splicing+p_deg), size=1)[0]
#         t+= stats.poisson.rvs(mu = (p_txn+p_splicing+p_deg), size=1)[0]

#     return(N,M)
    
# num_trajectories = 10000
# N_array = np.ones(num_trajectories)
# M_array = np.ones(num_trajectories)

# for i in range(num_trajectories):
#     print(f'Trajectory {i}')
#     N,M = simulate_one_trajectory(b,beta,gamma,stopping_time = 1000)
#     N_array[i] = N
#     M_array[i] = M

