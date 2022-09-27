# import necessary packages

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


    
# define model with varying max mean and max var
class MLP_1NB_varmax(nn.Module):
    def __init__(self, input_size, num_hidden_units, num_hidden_layers, output_size = 2, activate='relu',
                final_activation = 'sigmoid',max_mean=torch.tensor(2.),max_var=torch.tensor(2.)):
        super().__init__()
        self.activate = activate
        self.softplus = nn.Softplus()
        self.module_list = nn.ModuleList([])
        self.module_list.append(nn.Linear(input_size,num_hidden_units))
        for k in range(num_hidden_layers-1):
            self.module_list.append(nn.Linear(num_hidden_units, num_hidden_units))
        self.module_list.append(nn.Linear(num_hidden_units,output_size))
        self.final_activation = final_activation
        self.max_mean = nn.Parameter(max_mean,requires_grad=True)
        self.max_var = nn.Parameter(max_var,requires_grad=True)


    def forward(self, x):
        for f in self.module_list[:-1]:
            x = f(x)
            if self.activate == 'relu':
                x = F.relu(x)
            elif self.activate == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.activate == 'sin':
                x = torch.sin(x)
        
        if self.final_activation == 'sigmoid':
            out = torch.sigmoid(self.module_list[-1](x))
            C_mean = out[:,1]
            C_var = out[:,0]
            s_mean = C_mean*(self.max_mean-(1/self.max_mean)) + (1/self.max_mean)
            s_var = C_var*(self.max_var-(1/self.max_var)) + (1/self.max_var)
            
        elif self.final_activation == 'relu':
            out = F.relu(self.module_list[-1](x))
            print('activation is relu')
            s_mean = out[:,1]
            s_var = out[:,0]
            
        elif self.final_activation == 'softplus':
            print('activation is softplus')
            out = F.softplus(self.module_list[-1](x))
            s_mean = out[:,1]
            s_var = out[:,0]
            
        elif self.final_activation == 'none':
            print('activation is none')
            out = self.module_list[-1](x)
            s_mean = out[:,1]
            s_var = out[:,0]
     


        return s_mean,s_var
    
    
def train_MLP(model_config, train_config, train_set, valid_set):
    input_size = model_config['input_size']
    output_size = model_config['output_size']
    num_hidden_units = model_config['num_hidden_units']
    num_hidden_layers = model_config['num_hidden_layers']
    activate = model_config['activate']
    final_activation = model_config['final_activation']
    max_type = model_config['max_type']
  
    lr = train_config['lr']
    num_epochs = train_config['num_epochs']
    batchsize = train_config['batchsize']
    weight_decay = train_config['weight_decay']
    loss_type = train_config['loss_type']
    train_sin = train_config['train_sin']
    max_mean = train_config['max_mean']
    max_var = train_config['max_var']
    
  
    trials = int(len(train_set)/batchsize)
    
    if max_type == 'const':
        model = MLP_1NB(input_size, num_hidden_units, num_hidden_layers, output_size, activate,
                    final_activation = final_activation,
                   max_mean = max_mean,
                   max_var = max_var)
    if max_type == 'var':
        model = MLP_1NB_varmax(input_size, num_hidden_units, num_hidden_layers, output_size, activate,
                    final_activation = final_activation,
                   max_mean = max_mean,
                   max_var = max_var)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay) 
  
    valid_loss = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    train_true_prob = train_set[:,1]
    vecs = torch.stack([ torch.tensor(train_set[i][0],dtype=torch.float) for i in range(len(train_set))])
    
    if train_sin == 'True':
        print('train on sin of parameters')
        train_vecs = torch.sin(vecs)
    else:
        train_vecs = vecs
  
    for e in range(num_epochs):
        print('Epoch: ', e+1)
   
        # set up model 
        model.train()
    
        # shuffle data (indeces)
        idx = torch.randperm(len(train_set))
        vecs_shuffled = train_vecs[idx]
        train_vecs_shuffled = train_vecs[idx]
      
        train_true_prob_shuffled = np.take(train_true_prob, idx)
    
        # set up array to store losses
        trial_loss = np.zeros(trials)
    
        for i in range(trials):
            trial_vecs = vecs_shuffled[batchsize*i:batchsize*i+batchsize]
            trial_vecs_to_train = train_vecs_shuffled[batchsize*i:batchsize*i+batchsize]
            trial_true_prob = train_true_prob_shuffled[batchsize*i:batchsize*i+batchsize]
      
            optimizer.zero_grad()
      
            s_mean, s_var = model(trial_vecs_to_train)
      
            loss = loss_function(trial_true_prob,trial_vecs,s_mean,s_var,loss_type=loss_type)
      
            trial_loss[i] = loss.item()
            loss.backward()

      
            optimizer.step()
      
            train_loss[e] = np.mean(trial_loss)
            
    
        #check on validation data
        model.eval()
        valid_loss[e] = get_valid_metric(model,valid_set,loss_type=loss_type,train_sin=train_sin)
    

    return train_loss, valid_loss, model



EPS = 1e-20
eps = 1e-20

lnfactorial = torch.special.gammaln(torch.arange(1003))

def loss_function(trial_true_prob,trial_vecs,s_mean,s_var,return_y = False,use_old = False,
                  loss_type = 'KLD'):
  
    
    KL = torch.zeros(len(s_mean))
    MSE_array = torch.zeros(len(s_mean))

    b =  10**trial_vecs[:,0]
    beta =  10**trial_vecs[:,1]
    gamma =  10**trial_vecs[:,2]
    n = trial_vecs[:,3]
    mean_cond = trial_vecs[:,4]
    var_cond = trial_vecs[:,5]


    mean_cond_scaled = s_mean*mean_cond 
    var_cond_scaled = s_var*var_cond 

    for t in range(len(s_mean)):
        mean_cond_scaled_ = mean_cond_scaled[t]
        var_cond_scaled_ = var_cond_scaled[t]

        mean_cond_ = mean_cond[t]
        var_cond_= var_cond[t]
        beta_ = beta[t]

        mean_cond_ = mean_cond[t]
        var_cond_ = var_cond[t]


        true_prob_ = torch.tensor(trial_true_prob[t],dtype=torch.float)

        m = torch.arange(len(true_prob_),dtype=torch.float,requires_grad=True)

        if use_old == True:
            log_y = log_P(m,mean_cond_,var_cond_,beta_)
        else: 
            log_y = log_P(m,mean_cond_scaled_,var_cond_scaled_,beta_)

        y_ = torch.exp(log_y)

        if loss_type == 'MSE':
            MSE_array[t] = torch.mean((y_ - true_prob_)**2 )

        true_prob_norm_ = true_prob_/true_prob_.sum()

        # add small number so that no value is 0
        y_ = y_ + eps

        y_norm_ = (y_)/y_.sum()

        KL_ = -torch.sum(true_prob_norm_*torch.log(y_norm_/true_prob_norm_))
        #KL[t] = KL_/(len(true_prob_))
        KL[t] = KL_
    
    if return_y == True:
        return(torch.mean(KL),y_norm_,y_)

    if loss_type == 'MSE':
        return(torch.sum(MSE_array))

    else:
        return(torch.mean(KL))


def get_valid_metric(model,valid_set,loss_type='KLD',train_sin='False'):
    
    valid_true_prob = [ torch.tensor(valid_set[i,1],dtype=torch.float) for i in range(len(valid_set))]
    valid_vecs =  torch.stack([ torch.tensor(valid_set[i][0],dtype=torch.float) for i in range(len(valid_set))])

    
    if train_sin == 'True':
        valid_vecs_train = torch.sin(valid_vecs)
    else:
        valid_vecs_train = valid_vecs

    s_mean, s_var = model(valid_vecs_train)

    loss = loss_function(valid_true_prob,valid_vecs,s_mean,s_var,loss_type=loss_type)

    return loss


def log_P(m,mean_cond,var_cond,beta):   
    ''' Returns the LOG of the negative binomial probability given mean and variance at point (or points) m.
    '''
    r_cond = mean_cond**2/(var_cond-mean_cond)
    p_cond = mean_cond/var_cond
    r = 1/beta
   
    y_ = m * torch.log(mean_cond+eps) - mean_cond - torch.lgamma(m+1) 

    if torch.logical_and(torch.logical_and(r>0,p_cond>0), p_cond<1):
        y_ += torch.lgamma(m + r_cond) - torch.lgamma(r_cond) + r_cond * torch.log(r_cond/(r_cond+mean_cond)+eps) - m *          torch.log(r_cond +mean_cond+eps) + mean_cond 
   
    P =  y_
   
    if torch.any(~torch.isfinite(y_)): 
        print(y_)
        raise ValueError('bad y_')

   
    return P



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
    
def MMNB(p,n,m):
    
    p = torch.tensor(10**p)
    MU, VAR, STD, xmax = [torch.tensor(x) for x in get_moments(p)]
    
    COV = p[0]**2/(p[1]+p[2])
    n = torch.tensor(n)
    m = torch.tensor(m)
    
    logvar = torch.log((VAR/MU**2)+1)
    logstd = torch.sqrt(logvar)
    logmean = torch.log(MU**2/torch.sqrt(VAR+MU**2))

    logcov = torch.log(COV * torch.exp(-(logmean.sum()+logvar.sum()/2)) +1 ) 
    logcorr = logcov/torch.sqrt(logvar.prod())

    logmean_cond = logmean[1] + logcorr * logstd[1]/logstd[0] * (torch.log(n+1) - logmean[0])
    logvar_cond = logvar[1] * (1-logcorr**2)   

    mean_cond = torch.exp(logmean_cond + logvar_cond/2)
    var_cond = torch.exp(2*logmean_cond + logvar_cond) * (torch.exp(logvar_cond) - 1)

    r = 1/p[1]
    r_cond = mean_cond**2/(var_cond-mean_cond)
    p_cond = mean_cond/var_cond
    prefactor = torch.lgamma(n+r) - torch.lgamma(n+1) - torch.lgamma(r) \
                + r * torch.log(r/(r+MU[0])) + n * torch.log(MU[0]/(r+MU[0]))

    y_ = m * torch.log(mean_cond) - mean_cond - torch.lgamma(m+1) 
    filt = torch.logical_and(torch.logical_and(r>0,p_cond>0), p_cond<1)
    

    if filt == True:
        y_ += torch.lgamma(m+r_cond)  - torch.lgamma(r_cond) \
                + r_cond * torch.log(r_cond/(r_cond+mean_cond)) \
                - m * torch.log(r_cond+mean_cond) + mean_cond   

#     y_[filt] += torch.lgamma(m[filt]+r_cond[filt])  - torch.lgamma(r_cond[filt]) \
#                 + r_cond[filt] * torch.log(r_cond[filt]/(r_cond[filt]+mean_cond[filt])) \
#                 - m[filt] * torch.log(r_cond[filt]+mean_cond[filt]) + mean_cond[filt]

    P = prefactor +  y_

    return np.exp(P)


def calculate_metric(data,model,loss_type,use_old = False):

    model.train()
    metrics = np.zeros(len(data))

    for i in range(len(data)):
        true_prob_ = torch.tensor([data[i:i+1][0][1]])
        vec_ =  torch.tensor(data[i:i+1][0][0].reshape(1,-1),dtype=torch.float)

        s_mean, s_var = model(vec_)
    
        metric = loss_function(true_prob_,vec_,s_mean,s_var,return_y = False,use_old = use_old,
                  loss_type = 'KLD')
            
        metrics[i] = metric

    return(metrics)