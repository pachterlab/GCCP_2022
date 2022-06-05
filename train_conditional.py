# load in modules

import random
import numpy as np
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

import ypred_module as ypm

# load in data
def load_data(number_of_files,file_path,name):
    '''Loads in data files given file path and name of files (un-numbered).
      Returns list of numpy arrays of data.'''

    data_list = []
    
    for i in range(number_of_files):
        data_ = list(np.load(file_path+name+f'_{i}.npy',allow_pickle=True))
        data_list = data_list + data_

    return(data_list) 

# unpack data
def unpack_data(data_list):
    '''Load .npy file, returns tensor for parameters and ground truth histograms'''
 
    ps = np.array([ a[0] for a in data_list ])
    p_tensor = torch.from_numpy(ps).float()
    y_tensor = [ torch.tensor(a[1]).float() for a in data_list ]
    
    return(p_tensor,y_tensor)


# shuffle data 
def shuffle_data(data_list):
    '''shuffles the pre-loaded data list (keeps param vectors with y values)'''
   
    random.shuffle(data_list)
    parameters = np.array([ a[0] for a in data_list ])
    parameters_tensor = torch.tensor(parameters).float()
    y_tensor = [ torch.tensor(a[1]).float() for a in data_list ]
    
    return(parameters_tensor,y_tensor)

# get moments
def get_moments(p):

    b,beta,gamma=p
    
    r = torch.tensor([1/beta, 1/gamma])
    MU = b*r
    VAR = MU*torch.tensor([1+b,1+b*beta/(beta+gamma)])
    STD = torch.sqrt(VAR)
    xmax = torch.ceil(MU)
    xmax = torch.ceil(xmax + 4*STD)
    xmax = torch.clip(xmax,30,np.inf).int()
    return MU, VAR, STD, xmax

# get metrics
def get_metrics(ypred,y,metric):
    '''Calculates desired metric between predicted probability and y.'''
    
    y = torch.flatten(y)/y.sum()
    ypred = torch.flatten(ypred)/ypred.sum()
   

    if metric=='kld':
        return -torch.sum(y*torch.log(ypred/y))
    if metric=='kld_normalized':
        return -torch.sum(y*torch.log(ypred/y))/y.size(0)
    if metric=='totalse':
        return torch.sum((ypred-y)**2)
    if metric=='mse':
        return torch.mean((ypred-y)**2)
    if metric=='maxabsdev':
        return torch.max(torch.abs(ypred-y))
    if metric=='maxabsdevlog':
        return torch.max(torch.abs(torch.log(ypred)-torch.log(y)))
    if metric=='mselog':
        return torch.mean((torch.log(ypred)-torch.log(y))**2)



def calculate_test_metrics(test_list,model,get_ypred_at_RT,metric):
    ''' Calculates metric for a test_list given model and a function to generate kernel functions. 
    '''
    model.eval()
    p_list,y_list = unpack_data(test_list)
    metrics = np.zeros(len(p_list))
    
    for i in range(len(p_list)):
        y = y_list[i].flatten()
        ypred = get_predicted_PMF(p_list,i,model,get_ypred_at_RT)

        metric_ = get_metrics(ypred,y,metric)
        metrics[i] = metric_.detach().numpy()
        
    return(metrics,np.mean(metrics))

# get predicted PMF
def get_predicted_PMF(p_list,position,model,get_ypred_at_RT):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()
    p_ = p_list[position:position+1]
    
    w_,hyp_= model(p_)

    p = p_
    w = w_
    hyp = hyp_
    
    ypred = get_ypred_at_RT(p,w,hyp)
    
    return ypred 


# define loss function
def loss_fn(ps,ys,w,hyp,get_ypred_at_RT,metric):
    '''Calculates average metval over batch between predicted Y and y.
    yker_list and y_list are actually lists of tensor histograms with first dimension batchsize'''
    
    batchsize = len(ps)

    metval = torch.tensor(0.0)
 

    for b in range(batchsize):
       
        y_ = ys[b]
        p_ = ps[b:b+1]
        w_ = w[b:b+1]
        hyp_ = hyp[b:b+1]
        
        
        ypred_ = get_ypred_at_RT(p_,w_,hyp_)
      
        met_ = get_metrics(ypred_,y_,metric)
 
        metval += met_


    return metval/batchsize

# define model
class MLP(nn.Module):

    def __init__(self, input_dim, npdf, h1_dim, h2_dim):
        super().__init__()

        self.input = nn.Linear(input_dim, h1_dim)
        self.hidden = nn.Linear(h1_dim, h2_dim)
        self.output = nn.Linear(h2_dim, npdf)

        self.hyp = nn.Linear(h1_dim,1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = torch.sigmoid
        

    def forward(self, inputs):

        # pass inputs to first layer, apply sigmoid
        l_1 = self.sigmoid(self.input(inputs))

        # pass to second layer, apply sigmoid
        l_2 = self.sigmoid(self.hidden(l_1))
        
        # pass to output layer 
        w_un = (self.output(l_2))
        
        # pass out hyperparameter, sigmoid so it is between 0 and 1, then scale between 1 and 6
        hyp = self.sigmoid(self.hyp(l_2))
    
        # apply softmax
        w_pred = self.softmax(w_un)

        return w_pred,hyp



def run_epoch(p_list,y_list,model,optimizer,batchsize,get_ypred_at_RT,metric):

    model.train()

    # number of batches (data/batchsize)
    trials = int(np.floor(len(p_list) / batchsize ))

    metvals = torch.zeros(trials)

    for j in range(trials):
        i = j * batchsize
        ps = p_list[i:i+batchsize]
        ys = y_list[i:i+batchsize]

        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        w, hyp  = model(ps)

        # Compute loss
        loss = loss_fn(ps,ys,w,hyp,get_ypred_at_RT,metric)
        
        # average metric for the batch j
        metvals[j] = loss.item()
   
        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

            
    # calculate the average metric over the epoch 
    av_metval = torch.mean(metvals)

    return(av_metval)


def train(train_list,valid_list,model_config,train_config):
    

    # define model configurations
    npdf = model_config['npdf']
    input_dim = model_config['input_dim']
    h1_dim = model_config['h1_dim']
    h2_dim = model_config['h2_dim']
    

    # define model
    model = MLP(input_dim=input_dim, npdf=npdf,
     h1_dim=h1_dim, h2_dim=h2_dim)
        

    # define training configurations
    num_epochs = train_config['num_epochs']
    lr = train_config['lr']
    batchsize = train_config['batchsize']
    metric = train_config['metric']
    
    # define y pred function and grid based on the number of npdf -- defalut quantile spacing is Chebyshev
    NORM = ypm.get_NORM(npdf=npdf)
    
    # n range and m range are place holders but will be changed to propoer training when training = True
    get_ypred_at_RT = lambda p,w,hyp: ypm.get_ypred_at_RT(p,w,hyp,n_range=0,m_range=0,norm=NORM,training=True)

    # uss Adam optimizer with learning rate of lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    # store metric values for training and evaluation data
    train_metvals = np.zeros(num_epochs)
    valid_metvals = np.zeros(num_epochs)
    
    for e in range(num_epochs):
        print('Epoch Number:',e+1)

        # shuffle data
        p_list,y_list = shuffle_data(train_list)

        # run one epoch
        metval_ = run_epoch(p_list,y_list,model,optimizer,batchsize,get_ypred_at_RT,metric)
       
        # store epoch metric
        train_metvals[e] = metval_

        # test by evaluating the model
        valid_metval_list_,valid_metval_ = calculate_test_metrics(valid_list,model,get_ypred_at_RT,metric)
        
        # store test metric
        valid_metvals[e] = valid_metval_
        
        print(f'Train metric: {metval_}')
        print(f'Valid metric: {valid_metval_}')
    

    return(model, train_metvals, valid_metvals)