import numpy as np
import torch


import ypred_module as ypm
import train_conditional as train
import tools_conditional as tools


# Figure 3c,3d,3e: calculate testing KLDs for conditional testing parameters
path1= './models/'

test_list = train.load_data(3,'./data/','256_test')

best_model = tools.Trained_Model(path1, 'best_model')

# average calculated KLD of 3 test files with varying number of parameter example training files

params = np.array([1,2,3,4,5])
param_save = []


NORM = ypm.get_NORM(npdf=10)
get_ypred_at_RT = lambda p,w,hyp: ypm.get_ypred_at_RT(p,w,hyp,n_range=0,m_range=0,norm=NORM,training=True)

for p in params:
    print('params ',p)
    model = tools.Trained_Model(path1, f'{p}train')
    metrics,metric_mean = train.calculate_test_metrics(test_list,model.model,get_ypred_at_RT,metric='kld')
    param_save.append((256*p,metrics))

metrics,metric_mean = train.calculate_test_metrics(test_list,best_model.model,get_ypred_at_RT,metric='kld')
param_save.append((6*256,metrics))

param_save = np.array(param_save)
np.save('./results/testing_klds_params',param_save)

# now, for varying number of hidden nodes
hids = [1,32,64,128]
hid_save = []

for h in hids:
    print('hidden ',h)
    model = ttc.Trained_Model(path1, f'{h}nodes')
    metrics,metric_mean = train.calculate_test_metrics(test_list,model.model,get_ypred_at_RT,metric='kld')
    hid_save.append((h,metrics))

hid_save = np.array(hid_save)
np.save('./results/testing_klds_hid',hid_save)


# finally, for varying number of npdfs
npdfs = np.array([1,5,7,15,20])
npdf_save = []

for n in npdfs:
    print('npdf: ',n)
    NORM_ = ypm.get_NORM(npdf=n,quantiles='cheb')
    GET_YPRED = lambda p,w,hyp : ypm.get_ypred_at_RT(p,w,hyp,n_range=0,m_range=0,NORM=NORM_,training=True)
    model = tools.Trained_Model(path1, f'{n}npdf')
    metrics,metric_mean = train.calculate_test_metrics(test_list,model.model,GET_YPRED,metric='kld')
    npdf_save.append((n,metrics))


npdf_save = np.array(npdf_save)
np.save('./results/testing_klds_npdfs',npdf_save)