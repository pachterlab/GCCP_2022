# argument parser
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str)
parser.add_argument('--npdf', type=int)
parser.add_argument('--num_units', type=int, default = 128)
parser.add_argument('--num_layers', type=int, default = 2)
parser.add_argument('--num_training_files', type=int, default = 5)
parser.add_argument('--num_training_params', type=int, default = 256)
parser.add_argument('--num_epochs', type=int, default = 50)
parser.add_argument('--batchsize', type=int, default = 10)
parser.add_argument('--lr', type=float, default = 1e-4)
parser.add_argument('--scale_or_hyp', type=str, default = 'scale')
parser.add_argument('--max_mv', type=str, default = 'no_update')
parser.add_argument('--loss_function', type=str, default = 'kld')
# parser.add_argument('--max_mv', type=int, default = 0)

# parser.add_argument('--save_dir', type=str, default = '../data/')
# parser.add_argument('--data_dir', type=str, default = '../data/')
args = parser.parse_args()

name = args.name
npdf = args.npdf
num_units = args.num_units
num_layers = args.num_layers
num_training_files = args.num_training_files
num_epochs = args.num_epochs
batchsize = args.batchsize
lr = args.lr
num_training_params = args.num_training_params
scale_or_hyp = args.scale_or_hyp
max_mv = args.max_mv
loss_function = args.loss_function

# modules
import os
import time

# math
import numpy as np
import torch
import trainers_2 as tm2

import pickle


# directories!
import os
import shutil


max_val = 5.0

model_plan = { 'input_size' : 4+2*npdf,
               'npdf' : npdf,
               'num_units' : num_units,
               'num_layers' : num_layers,
               'activate' : 'relu',
               'final_activation' : 'sigmoid',
               'max_mv' : max_mv,
               'max_val' : max_val,
               'type' : 'learn_weights_' + scale_or_hyp
            }


training_plan = {
    'lr' : lr,
    'batchsize' : batchsize,
    'weight_decay' : 1e-9,
    'n_epochs' : num_epochs,
    'loss_function' : loss_function
}

model = tm2.Trainer(model_plan)


train_ds = tm2.burstyBernoulliDataset([f'../data/{npdf}_npdf_cheb/{num_training_params}_params_{i}_train' for i in range(num_training_files)])
valid_ds = tm2.burstyBernoulliDataset([f'../data/{npdf}_npdf_cheb/{num_training_params}_params_{i}_valid' for i in range(1)])


torch.autograd.set_detect_anomaly(True)
t1 = time.time()
save_dict = model.train(train_ds,valid_ds,training_plan)
t2 = time.time()


save_dict['model_plan'] = model_plan
save_dict['num_training_params'] = num_training_params
save_dict['training_plan'] = training_plan
save_dict['total_time'] = t2 - t1

if loss_function == 'kld':
    path = f'../results/learn_weights_{scale_or_hyp}/{npdf}npdf_{num_layers}hl_{num_units}hu_{batchsize}bs_{lr}lr_{max_mv}_{num_training_params}/'
    
elif loss_function == 'mse':
        path = f'../results/learn_weights_{scale_or_hyp}_{loss_function}/{npdf}npdf_{num_layers}hl_{num_units}hu_{batchsize}bs_{lr}lr_{max_mv}_{num_training_params}/'
        
        
elif loss_function == 'hellinger':
        path = f'../results/learn_weights_{scale_or_hyp}_{loss_function}/{npdf}npdf_{num_layers}hl_{num_units}hu_{batchsize}bs_{lr}lr_{max_mv}_{num_training_params}/'
        
elif loss_function == 'joint_hellinger':
        path = f'../results/learn_weights_{scale_or_hyp}_{loss_function}/{npdf}npdf_{num_layers}hl_{num_units}hu_{batchsize}bs_{lr}lr_{max_mv}_{num_training_params}/'

elif loss_function == 'joint_mse':
        path = f'../results/learn_weights_{scale_or_hyp}_{loss_function}/{npdf}npdf_{num_layers}hl_{num_units}hu_{batchsize}bs_{lr}lr_{max_mv}_{num_training_params}/'

if not os.path.exists(path):
    os.makedirs(path)
else:
    shutil.rmtree(path)           
    os.makedirs(path)


print(path)

# !ls ../results/{num_hidden_layers}hl_{num_hidden_units}hu_{num_training_files}train_{batchsize}bs_2max/
torch.save(model,path+'MODEL')

# check performance on test data!! 
test_ds = tm2.burstyBernoulliDataset([f'../data/{npdf}_npdf_cheb/{num_training_params}_params_{i}_test' 
                                        for i in range(1)])
params = test_ds.get_data_dict(None)['params']
cond = test_ds.get_data_dict(None)['cond_pss'].numpy()
N = len(params)
eps = 1e-18
kld_nn = np.ones(N)
kld_mmnb = np.ones(N)
hell_nn = np.ones(N)
hell_mmnb = np.ones(N)


m_limit = 300
m = torch.arange(m_limit)
weights = torch.ones((len(params),npdf))*(1/npdf)


# mmnb
cond_means = params[:,4:4+npdf]
cond_stds = params[:,4+npdf:]
m = torch.arange(m_limit)              
mmnb = model.eval_cond_P(m,weights,cond_means,cond_stds).detach().numpy()
 
# predicted
pred = model.get_cond_P(params, m_limit).detach().numpy()


# normalize
cond = cond + eps
pred = pred + eps
mmnb = mmnb + eps
cond = cond/ (cond.sum(axis=1)[:,None])
pred = pred/ (pred.sum(axis=1)[:,None])
mmnb = mmnb/ (mmnb.sum(axis=1)[:,None])


hell_nn[:] = 0.5 * np.sum( (np.sqrt(cond) - np.sqrt(pred))**2, axis = 1)
hell_mmnb[:] = 0.5 * np.sum( (np.sqrt(cond) - np.sqrt(mmnb))**2, axis = 1)
    
kld_nn[:] = np.sum(cond * np.log(cond/pred), axis = 1)
kld_mmnb[:] = np.sum(cond * np.log(cond/mmnb), axis = 1)
    
    
# for i in range(N):
#     p_ = params[i:i+1]
#     cond_ = cond_pss[i:i+1].detach().numpy().flatten()
#     m_limit = 300
#     pred_ = model.get_cond_P(p_, m_limit).detach().numpy().flatten()
    
#     weights = (torch.ones(npdf)*(1/npdf))[None,:]
#     cond_means = p_[:,4:4+npdf]
#     cond_stds = p_[:,4+npdf:]
#     m = torch.arange(m_limit)              
#     mmnb_ = model.eval_cond_P(m,weights,cond_means,cond_stds).detach().numpy().flatten()
    
#     pred_ = pred_ + eps
#     cond_ = cond_ + eps
#     mmnb_ = mmnb_ + eps
    
    
#     cond_ = cond_/np.sum(cond_)
    
#     hell_nn[i] = 0.5 * np.sum( (np.sqrt(cond_) - np.sqrt(pred_))**2 )
#     hell_mmnb[i] = 0.5 * np.sum( (np.sqrt(cond_) - np.sqrt(mmnb_))**2 )
    
#     pred_ = pred_/ np.sum(pred_)
#     cond_ = cond_/ np.sum(cond_)
#     mmnb_ = mmnb_/ np.sum(mmnb_)
    
#     kld_nn[i] = np.sum(cond_ * np.log(cond_/pred_))
#     kld_mmnb[i] = np.sum(cond_ * np.log(cond_/mmnb_))
    
save_dict['test_kld_nn'] = kld_nn
save_dict['test_kld_mmnb'] = kld_mmnb
save_dict['test_hellinger_nn'] = hell_nn
save_dict['test_hellinger_mmnb'] = hell_mmnb

# Store data 
with open(path+'meta','wb') as handle:
    pickle.dump(save_dict, handle)