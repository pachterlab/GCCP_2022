# import modules

import numpy as np

import torch

import train_conditional as train
import ypred_module as ypm
import exact_cme as cme

import matplotlib.pyplot as plt


def hellinger(p,q):
    
    p_sqrt = np.sqrt(p)
    q_sqrt = np.sqrt(q)
    a = (p_sqrt-q_sqrt)**2
    b = np.sqrt(np.sum(a))
    
    return (1/(np.sqrt(2)))*b



def get_full_metrics(data_set,metric,model=ypm.model):
    
    length = len(data_set)
    metrics = np.zeros(length)
    
    for i in range(length):
        p_,pmf_ = data_set[i][0],data_set[i][1]
        pred_ = predict_full_pmf(p_,pmf_,model).flatten()
        pmf_ = pmf_.flatten()
        if metric== 'hellinger':
            metric_ = hellinger(pmf_,pred_)
        else:
            pred_ = pred_/pred_.sum()
            pmf_ = pmf_/pmf_.sum()
            metric_ = -np.sum(pmf_*np.log(pred_/pmf_))
        metrics[i] = metric_
        
    return metrics


def get_conditional_metrics(data_set,metric,model=ypm.model):
    
    length = len(data_set)
    metrics = np.zeros(length)
    
    for i in range(length):
        p_,pmf_ = data_set[i][0],data_set[i][1]
        pred_ = ypm.get_prob(p_[:3],n_range=np.array(p_[-1:]),m_range=np.arange(len(pmf_)),model=model,rand_weights=False)
        pmf_ = pmf_.flatten()
        if metric== 'hellinger':
            metric_ = hellinger(pmf_,pred_)
        else:
            pred_ = pred_/pred_.sum()
            pmf_ = pmf_/pmf_.sum()
            metric_ = -np.sum(pmf_*np.log(pred_/pmf_))
        metrics[i] = metric_
        
    return metrics

def plot_loss(train_metvals,valid_metvals,metric):
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(train_metvals,c='blue',label='Training')
    plt.plot(valid_metvals,c='red',label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend();




def xmax_fun(x):
    return x


def plot_pmf(p_vec,model = ypm.model,true_pmf = np.zeros(1),return_pmfs = False):
    
    if true_pmf.any():

        
        true_pmf_ = true_pmf
        pred_ = predict_full_pmf(p_vec,true_pmf_,model=model)
        fig1,ax1=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
        
        vmax = np.max(np.concatenate((pred_,true_pmf_)))
        vmin = np.min(np.concatenate((pred_,true_pmf_)))
        
        ax1[0].imshow(true_pmf_,aspect='auto',vmin=vmin,vmax=vmax)
        ax1[0].invert_yaxis()
        ax1[0].set_title('True Probability')
        ax1[0].set_xlabel('# mature RNA')
        ax1[0].set_ylabel('# nascent RNA')

        hell_ = hellinger(pred_.flatten(),true_pmf_.flatten())
        
        ax1[1].imshow(pred_,aspect='auto',vmin=vmin,vmax=vmax)
        ax1[1].invert_yaxis()
        ax1[1].set_title(f'Approximation, Hellinger Distance {hell_:.5f}')
        ax1[1].set_xlabel('# mature RNA')
        ax1[1].set_ylabel('# nascent RNA')
        plt.show();
        
        if return_pmfs == True:
            return(pred_,true_pmf_)
        
    else:
        p, true_pmf_ = cme.calculate_exact_cme(p_vec,method='quad_vec',xmax_fun=xmax_fun) 
        pred_ = predict_full_pmf(p_vec,true_pmf_,model=model)
        
        fig1,ax1 = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
        
        vmax = np.max(np.concatenate((pred_.flatten(),true_pmf_.flatten())))
        vmin = np.min(np.concatenate((pred_.flatten(),true_pmf_.flatten())))
                
        ax1[0].imshow(true_pmf_,aspect='auto',vmin=vmin,vmax=vmax)
        ax1[0].invert_yaxis()
        ax1[0].set_title('True Probability')
        ax1[0].set_xlabel('# mature RNA')
        ax1[0].set_ylabel('# nascent RNA')

        hell_ = hellinger(pred_.flatten(),true_pmf_.flatten())
        
        ax1[1].imshow(pred_,aspect='auto',vmin=vmin,vmax=vmax)
        ax1[1].invert_yaxis()
        ax1[1].set_title(f'Approximation, Hellinger Distance {hell_:.5f}')
        ax1[1].set_xlabel('# mature RNA')
        ax1[1].set_ylabel('# nascent RNA')
        plt.show();
        
        if return_pmfs == True:
            return(pred_,true_pmf_)


def plot_histogram(array,bins,metric='KLD'):
    '''Histogram of bin number of bins, xlim'''
    plt.hist(array,bins = bins)
    plt.title(metric)
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.show();


def plot_CDF(array,metric='KLD'):

    cdf = np.zeros(len(array))
    array_sorted = np.sort(array)
    for i,value in enumerate(array_sorted):
        cdf[i] = len(array_sorted[array_sorted<value])/len(array_sorted)

    plt.scatter(array_sorted,cdf,s=5)
    plt.title(metric+' CDF')
    plt.xlabel(f'{metric}')
    plt.ylabel('CDF')
       
    plt.show();


def predict_full_pmf(p_vec,true_pmf,model=ypm.model):

    nas_range = np.arange(true_pmf.shape[0])
    mat_range = np.arange(true_pmf.shape[1])

    predicted = ypm.get_prob(p_vec,nas_range,mat_range,rand_weights=False)
    
    return(predicted)

def get_parameters_quantile(train_list,metrics,quantiles = [.95,1.0]):
    '''Returns given percent parameters with the highest klds and klds.'''
    
    parameters,y_list = train.unpack_data(train_list)
    
    metric_low = np.quantile(metrics,quantiles[0])
    metric_high = np.quantile(metrics,quantiles[1])
    
    metrics_segment = metrics[metrics>metric_low]
    params_segment = parameters[metrics>metric_low]
    
    metrics_segment_ = metrics_segment[metrics_segment<metric_high]
    params_segment_ = params_segment[metrics_segment<metric_high]
    
    return(params_segment_,metrics_segment_)

def plot_param_quantiles(klds,train_list):
    
    params_segment_1,klds_segment_1 = get_parameters_quantile(train_list,klds,quantiles=[0,.25])
    params_segment_2,klds_segment_2 = get_parameters_quantile(train_list,klds,quantiles=[.25,.5])
    params_segment_3,klds_segment_3 = get_parameters_quantile(train_list,klds,quantiles=[.5,.75])
    params_segment_4,klds_segment_4 = get_parameters_quantile(train_list,klds,quantiles=[.75,.95])
    params_segment_5,klds_segment_5 = get_parameters_quantile(train_list,klds,quantiles=[.95,1.])
    
    b_1 = 10**np.array([ p[0] for p in params_segment_1 ])
    beta_1 = 10**np.array([ p[1] for p in params_segment_1  ])
    gamma_1 = 10**np.array([ p[2] for p in params_segment_1  ])

    b_2 = 10**np.array([ p[0] for p in params_segment_2 ])
    beta_2 = 10**np.array([ p[1] for p in params_segment_2  ])
    gamma_2 = 10**np.array([ p[2] for p in params_segment_2  ])

    b_3 = 10**np.array([ p[0] for p in params_segment_3 ])
    beta_3 = 10**np.array([ p[1] for p in params_segment_3  ])
    gamma_3 = 10**np.array([ p[2] for p in params_segment_3  ])

    b_4 = 10**np.array([ p[0] for p in params_segment_4 ])
    beta_4 = 10**np.array([ p[1] for p in params_segment_4  ])
    gamma_4 = 10**np.array([ p[2] for p in params_segment_4  ])

    b_5 = 10**np.array([ p[0] for p in params_segment_5 ])
    beta_5 = 10**np.array([ p[1] for p in params_segment_5  ])
    gamma_5 = 10**np.array([ p[2] for p in params_segment_5  ])
    
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,5))
    
    
    # some labels
    ax[0].scatter(10,10,c='grey',label = 'Quantile 0-0.25')
    ax[0].scatter(10,10,c='blue',label = 'Quantile 0.25-0.50')
    ax[0].scatter(10,10,c='purple',label = 'Quantile 0.50-0.75')
    ax[0].scatter(10,10,c='green',label = 'Quantile 0.75-0.95')
    ax[0].scatter(10,10,c='red',label = 'Quantile 0.95-1.0')

    ax[0].scatter(b_1,beta_1,c = klds_segment_1, cmap= 'Greys')
    ax[0].scatter(b_2,beta_2,c = klds_segment_2, cmap= 'Blues')
    ax[0].scatter(b_3,beta_3, c = klds_segment_3, cmap= 'Purples')
    ax[0].scatter(b_4,beta_4,c = klds_segment_4, cmap= 'Greens')
    ax[0].scatter(b_5,beta_5,c = klds_segment_5, cmap= 'Reds')
    ax[0].set_xlabel('b')
    ax[0].set_ylabel('beta')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[1].scatter(b_1,gamma_1,c = klds_segment_1, cmap= 'Greys')
    ax[1].scatter(b_2,gamma_2, c = klds_segment_2, cmap= 'Blues')
    ax[1].scatter(b_3,gamma_3,c = klds_segment_3, cmap= 'Purples')
    ax[1].scatter(b_4,gamma_4,c = klds_segment_4, cmap= 'Greens')
    ax[1].scatter(b_5,gamma_5,c = klds_segment_5, cmap= 'Reds')
    ax[1].set_xlabel('b')
    ax[1].set_ylabel('gamma')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[2].scatter(beta_1,gamma_1,c = klds_segment_1, cmap= 'Greys')
    ax[2].scatter(beta_2,gamma_2, c = klds_segment_2, cmap= 'Blues')
    ax[2].scatter(beta_3,gamma_3,c = klds_segment_3, cmap= 'Purples')
    ax[2].scatter(beta_4,gamma_4,c = klds_segment_4, cmap= 'Greens')
    ax[2].scatter(beta_5,gamma_5,c = klds_segment_5, cmap= 'Reds')
    ax[2].set_xlabel('beta')
    ax[2].set_ylabel('gama')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
 
    ax[0].legend()
    fig.tight_layout()
    plt.title('MLP 1 Parameters Colored by KLD Quantile')



def save_model_and_meta(model,model_config,train_config,train_loss,valid_loss,time,path,name):
    
    torch.save(model.state_dict(),path+name+'_MODEL')

    meta = np.array([model_config,train_config,e,t,time])
    
    np.save(path+name+'_meta',meta)


class Trained_Model():
    
    def __init__(self, path, name):
        
        meta = np.load(path + name + '_meta.npy',allow_pickle=True)
        self.model_config = meta[0]
        self.train_config = meta[1]
        self.train_loss = meta[2]
        self.valid_loss = meta[3]
        self.time = meta[4]
        print(self.model_config)
        
        self.model = train.MLP(self.model_config['input_dim'],
                             self.model_config['npdf'], 
                             self.model_config['h1_dim'], 
                             self.model_config['h2_dim'])
            
        self.model.load_state_dict(torch.load(path+name+'_MODEL'))
        self.model.eval()


