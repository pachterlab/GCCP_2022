# trainers for speeding up process

# math
import numpy as np
np.random.seed(45)

import random
random.seed(45)

# pytorch
import torch
torch.manual_seed(45)
import torch.nn as nn
import torch.nn.functional as F

# iterables 
from collections.abc import Iterable

# miscellaneous 
import pickle
import pdb

# my models
from models import MLP_weights_hyp, MLP_weights_scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
eps = 1e-18


torch.set_printoptions(precision=10)

class burstyBernoulliDataset():
    ''' Dataset class for loading training data. 
    '''
    def __init__(self,src_file_list):
        data_list = []
        for src_file in src_file_list:
            file = open(src_file, 'rb')
            data = pickle.load(file)
            file.close()
            data_list = data_list+data
        
        self.data_list = data_list
        self.len = len(self.data_list)
        self.params = torch.tensor([data_list[i][0] for i in range(len(data_list))],dtype=torch.float32)
        self.cond_pss = torch.tensor([data_list[i][1] for i in range(len(data_list))],dtype=torch.float32)
        #self.cond_pss = self.cond_pss.reshape(-1,300)
    
    def get_data_dict(self,idx):
        
        if idx is not None:
            cond_pss = self.cond_pss[idx]
            parameters = self.params[idx]
      
            sample = {'cond_pss' : cond_pss, 'params' : parameters}
        
        else:
            cond_pss = self.cond_pss
            parameters = self.params
      
            sample = {'cond_pss' : cond_pss, 'params' : parameters}
        
        return sample
    
    def get_batch_idxs(self,batchsize,shuffle = True):
        
        idx = list(range(self.len))
        
        if shuffle == True:
            random.shuffle(idx)

        batch_idxs = []
        
        for i in range(int(np.ceil(self.len/batchsize))):
            
            batch_idxs.append(idx[i*batchsize:(i+1)*batchsize])
        
        return batch_idxs


def get_log_cond_moments(params):
    ''' Returns the logmean and logvar given a moment matched bivariate log-normal distribution.
    '''
    b = 10**params[:,0] 
    beta = 10**params[:,1]
    gamma = 10**params[:,2]
    n = params[:,-1]
    
    mu1 = b/beta
    mu2 = b/gamma

    var1 = (mu1)*(b + 1)
    var2 = (mu2)*(b*beta/(beta+gamma) + 1)
    
    cov = b**2/(beta+gamma)

    # moment match to lognormal
    logvar1 = torch.log((var1/mu1**2)+1)
    logvar2 = torch.log((var2/mu2**2)+1)
    logstd1 = torch.sqrt(logvar1)
    logstd2 = torch.sqrt(logvar2)

    logmean1 = torch.log(mu1**2/torch.sqrt(var1+mu1**2))
    logmean2 = torch.log(mu2**2/torch.sqrt(var2+mu2**2))

    logcov = torch.log(cov * torch.exp(-(logmean1 + logmean2 + (logvar1 + logvar2)/2)) +1 )
    logcorr = logcov/torch.sqrt(logvar1 * logvar2)
    
    # get conditional log moments 
    logmean_cond = logmean2 + logcorr * logstd2/logstd1 * (torch.log(n+1) - logmean1)
    logvar_cond = logvar2 * (1-logcorr**2)
    logstd_cond = logstd2 * torch.sqrt(1-logcorr**2)
    
    mean_cond = torch.exp(logmean_cond + logvar_cond/2)
    var_cond = torch.exp(2*logmean_cond + logvar_cond) * (torch.exp(logvar_cond) - 1)
    
    return mean_cond,var_cond



class Trainer():

    def __init__(self,model_plan):
        
        self.model_type = model_plan['type']
        self.npdf = model_plan['npdf']
        
        if model_plan['type'] == 'learn_weights_hyp':
            self.model = MLP_weights_hyp(input_size = model_plan['input_size'],
                            npdf = model_plan['npdf'],
                            num_layers = model_plan['num_layers'],
                            num_units = model_plan['num_units'],
                            activate = model_plan['activate']
                            )
        elif model_plan['type'] == 'learn_weights_scale':
            self.model = MLP_weights_scale(input_size = model_plan['input_size'],
                            npdf = model_plan['npdf'],
                            num_layers = model_plan['num_layers'],
                            num_units = model_plan['num_units'],
                            activate = model_plan['activate'],
                            final_activation = model_plan['final_activation'],
                            max_mv = model_plan['max_mv'],
                            max_val = model_plan['max_val'],
                            )
    
    def train(self,train_ds,valid_ds,training_plan):
        
        lr = training_plan['lr']
        weight_decay = training_plan['weight_decay']
        n_epochs = training_plan['n_epochs']
        batchsize = training_plan['batchsize']
        loss_function = training_plan['loss_function']
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay) 
        
        train_loss = np.ones(n_epochs)
        batch_idxs = train_ds.get_batch_idxs(batchsize,shuffle=True)
        train_loss_batch = np.zeros(n_epochs*len(batch_idxs[:-1]))
        valid_loss = np.ones(n_epochs)
        
        valid_dict = valid_ds.get_data_dict(None)
        valid_params = valid_dict['params']
        valid_cond_pss = valid_dict['cond_pss']
        
        for e in range(n_epochs):
            
            print(f'Starting epoch number: {e+1}')
            self.model.train()
            batch_idxs = train_ds.get_batch_idxs(batchsize,shuffle=True)
            batch_loss = np.ones(len(batch_idxs[:-1]))

            
            for i,idxs in enumerate(batch_idxs[:-1]):
                
                # get batch parameters and Pss
                batch_dict = train_ds.get_data_dict(idxs)
                params = batch_dict['params']
                cond_pss = batch_dict['cond_pss']
   
                # zero optimizer
                optimizer.zero_grad()
                
                weights, cond_means, cond_stds = self.model.forward(params)

                
                if torch.any(~torch.isfinite(weights)):
                    raise ValueError('the returned weights for NB are bad')
#                     pdb.set_trace()
#                     breakpoint()
                if torch.any(~torch.isfinite(cond_means)):
                    raise ValueError('the returned cond means for NB are bad')
#                     pdb.set_trace()
#                     breakpoint()
                if torch.any(~torch.isfinite(cond_stds)):
                    raise ValueError('the returned cond stds for NB are bad')
#                     pdb.set_trace()
#                     breakpoint()

#                for j,layer in enumerate(self.model.module_list):
#                     if torch.any(~torch.isfinite(layer.weight)):
#                         pdb.set_trace()
#                         breakpoint()


                loss = self.eval_cond_loss(weights,cond_means,cond_stds,cond_pss,params,loss_function,print_loss=False)
#                 print('weights after loss:',weights[0])
#                 print('cond_means after loss:',cond_means[0])
#                 print('cond_stds after loss:',cond_stds[0])
                
                batch_loss[i] = loss.item()
#                 print('batch loss',loss.item())
      
                loss.backward()
                
                for i,layer in enumerate(self.model.module_list):
                    if torch.any(~torch.isfinite(layer.weight.grad)):
                        raise ValueError(f'the gradients for {i} layer are bad')
#                         pdb.set_trace()
#                         breakpoint()
                    if torch.any(~torch.isfinite(layer.weight)):
                        raise ValueError(f'the weights for {i} layer are bad')
#                         pdb.set_trace()
#                         breakpoint()

                
                optimizer.step()

                   
                
                
            train_loss[e] = np.mean(batch_loss)


            train_loss_batch[int(e*len(batch_idxs[:-1])) : int((e+1)*len(batch_idxs[:-1]))] = batch_loss
            
            # get validation loss
            self.model.eval()
            valid_weights, valid_cond_means, valid_cond_stds = self.model.forward(valid_params)
            valid_loss_ = self.eval_cond_loss(valid_weights,
                                             valid_cond_means,
                                             valid_cond_stds,
                                             valid_cond_pss,
                                             valid_params,
                                             loss_function)

            valid_loss[e] = valid_loss_

#         return({'valid_loss' : valid_loss, 'train_loss' : train_loss, 'final_p' : params[0,:],
#                 'final_cond' : conds[0]})

        return({'valid_loss' : valid_loss, 'train_loss' : train_loss, 
               'train_loss_batch' : train_loss_batch})
    


    def eval_cond_P(self,m,weights,cond_means,cond_stds,print_loss=False):   
        ''' Returns the LOG of the negative binomial probability given means and variances at point (or points) m.
        '''

        p_nb = 1 - cond_means/cond_stds**2
        poisson_index = p_nb < 1e-4
        nb_index = p_nb >= 1e-4
        cond_stds[poisson_index] = torch.sqrt(cond_means[poisson_index]*1.05)
        cond_rs = (cond_means**2/(cond_stds**2-cond_means))
        
        filt2 = nb_index[:,:,None].repeat(1,1,len(m))


#         if filt2.sum() > 0:
#             print('using nb somewhere')
#         else:
#             print('NOT using NB ANYWHERE')

        if torch.any(~torch.isfinite(cond_stds)):
#             pdb.set_trace()
#             breakpoint()
#             pdb.set_trace()
#             breakpoint()
#             print('Bad y_ poisson')
#             print(y_)
#             print('R COND',cond_rs)
#             print('COND_MEAN',cond_means)
            raise ValueError('bad cond_std in EVAL_COND_P')
    
    
        if torch.any(~torch.isfinite(cond_means)):
#             pdb.set_trace()
#             breakpoint()
#             print('Bad y_ poisson')
#             print(y_)
#             print('R COND',cond_rs)
#             print('COND_MEAN',cond_means)
            raise ValueError('bad cond_means in EVAL_COND_P')
        
        if torch.any(~torch.isfinite(cond_rs)):
#             pdb.set_trace()
#             breakpoint()
#             print('Bad y_ poisson')
#             print(y_)
            print('R COND',cond_rs[nb_index])
#             print('COND_MEAN',cond_means)
            raise ValueError('bad cond rs')  
       
        #filt2 = filt1.repeat(1,1,len(m)).reshape(cond_rs.shape[0],len(m),cond_rs.shape[1])
        
        # for training purposes
#         cond_rs_nb = cond_rs[filt1].reshape(-1,3)[:,None]
#         cond_mean_nb = cond_means[filt1].reshape(-1,3)[:,None]
        
        # reshape
        cond_means = cond_means[:,:,None]
        cond_rs = cond_rs[:,:,None]
        weights = weights[:,:,None]
        
        
        # poisson
        y_ = m * torch.log(cond_means + eps) - cond_means - torch.lgamma(m+1)
        
        if torch.any(~torch.isfinite(y_)):
            print('bad y poisson')
            pdb.set_trace()
            breakpoint()
#             print('Bad y_ poisson')
#             print(y_)
#             print('R COND',cond_rs)
#             print('COND_MEAN',cond_means)


#         # negative binomial
#         a = torch.lgamma(m + cond_rs + eps)[filt2]
#         b = torch.lgamma(cond_rs + eps)[filt1.reshape(1000,1,3)]
#         if torch.any(~torch.isfinite(a)):
#             print('BAD A[filt2]')
#             pdb.set_trace()
#             breakpoint()
# #             print('Bad y_ poisson')
# #             print(y_)
# #             print('R COND',cond_rs)
# #             print('COND_MEAN',cond_means)
#             raise ValueError('bad y_ poisson') 
#         if torch.any(~torch.isfinite(b)):
#             print('BAD B[filt2]')
#             pdb.set_trace()
#             breakpoint()
#             print('Bad y_ poisson')
#             print(y_)
#             print('R COND',cond_rs)
#             print('COND_MEAN',cond_means)

#         step1 = torch.special.gammaln(m + cond_rs + eps)\
#                 - torch.special.gammaln(cond_rs + eps)\
#                 + m * torch.log(cond_means/(cond_rs + cond_means + eps) + eps)\
#                 + cond_rs * torch.log(cond_rs/(cond_rs + cond_means + eps))\
#                 - m * torch.log(cond_means + eps)\
#                 + cond_means

        step1 = torch.special.gammaln(m + cond_rs + eps)\
                - torch.special.gammaln(cond_rs + eps)\
                - m * torch.log(cond_rs + cond_means + eps)\
                + cond_means\
                + cond_rs * torch.log(cond_rs/(cond_rs + cond_means + eps) + eps)


#         if torch.any(~torch.isfinite(torch.special.gammaln(m + cond_rs))):
#             print('Bad torch.special.gammaln(m + cond_rs)')

#         if torch.any(~torch.isfinite(torch.special.gammaln(cond_rs))):
#             print('Bad torch.special.gammaln(cond_rs)')

#         if torch.any(~torch.isfinite(m * torch.log(cond_rs + cond_means + eps))):
#             print('Bad m * torch.log(cond_rs + cond_means + eps)')

#         if torch.any(~torch.isfinite(cond_rs * torch.log(cond_rs/(cond_rs + cond_means) + eps))):
#             print('Bad cond_rs * torch.log(cond_rs/(cond_rs + cond_means) + eps)')

            
            
        if torch.any(~torch.isfinite(step1)):
            print('Bad STEP')
            pdb.set_trace()
            breakpoint()


#         a = (torch.special.gammaln(m + cond_rs + eps) + torch.special.gammaln(cond_rs + eps))[filt2]
# #         print('torch.special.gammaln(m + cond_rs + eps)', a[a>1e10], a[a<1e-10])
# #         b = torch.special.gammaln(cond_rs + eps)[filt2]
#         print('torch.special.gammaln(m + cond_rs + eps)', a[torch.abs(a)>1e10], a[torch.abs(a)<1e-10])
#         if torch.any(~torch.isfinite(step[filt2])):
#             print('BAD STEP[filt2]')
# #             pdb.set_trace()
# #             breakpoint()
# #             print('Bad y_ poisson')
# #             print(y_)
#             print('R COND',cond_rs[filt2])
#             print('COND_MEAN',cond_means[filt2])
#             print(cond_stds[filt2])
#             raise ValueError('bad step[filt2]') 

#         step_test = torch.ones(cond_rs.shape[0],300,3)*eps
#         step_test[~filt2] = step_test[~filt2]/0

        y_[filt2] += step1[filt2] 
#         y_[filt2] += step
#         y_ += step1

        if torch.any(~torch.isfinite(weights)):
#             pdb.set_trace()
#             breakpoint()
            
            print('Bad WEIGHTS')
        
        
        y_ = torch.mul(weights,torch.exp(y_))
        
        if torch.any(~torch.isfinite(y_)):
#             pdb.set_trace()
#             breakpoint()
            
            print('Bad MUL Y')
        
        
        Y = y_.sum(axis=1)

        if torch.any(~torch.isfinite(Y)):
#             pdb.set_trace()
#             breakpoint()
            
            print('Bad Y NB post sum')
            pdb.set_trace()
            breakpoint()

#             print(y_)
#             print('cond_stds',cond_stds)
#             print('weights',weights)
#             print('cond_rs',cond_rs)
#             print('cond_means',cond_means)
#             print(filt)
#             raise ValueError('bad Y NB') 

        return(Y)
    

    def eval_cond_loss(self, weights, cond_means, cond_stds, cond_pss, params, loss_function = 'kld',
                      print_loss=False):
        
        '''Evaluate the predicted conditional probability given model returned cond_mean
        and cond_var (negative binomial). Then, evaluate KLD between predicted and returned.
        '''

#         if not isinstance(cond_pss, Iterable):
#             cond_pss = [cond_pss]

#         klds = 0
        cond = cond_pss
        m = torch.arange(300)

        pred = self.eval_cond_P(m,weights,cond_means,cond_stds,print_loss=print_loss)
 

        if torch.any(~torch.isfinite(pred)):
#             pdb.set_trace()
#             breakpoint()
            
            print('Bad PRED')
            pdb.set_trace()
            breakpoint()

#             pred[pred < eps] = eps
#             cond[cond < eps] = eps

        

        
        if loss_function == 'kld':
            cond[cond < eps] += (eps - cond[cond < eps])
            pred[pred < eps] += (eps - pred[pred < eps])
            cond = cond/ (cond.sum(axis=1)[:,None])
            pred = pred/ (pred.sum(axis=1)[:,None])
            if torch.any(~torch.isfinite(pred)):

                print('bad PRED post normalization',pred[~torch.isfinite(pred)])
                pdb.set_trace()
                breakpoint()
            
            loss = torch.sum(cond * (torch.log(cond/pred + eps)),axis=1)
            loss = torch.mean(loss)
            if print_loss == True:
                print(loss[0])


        
        elif loss_function == 'mse':
            cond[cond < eps] += (eps - cond[cond < eps])
            pred[pred < eps] += (eps - pred[pred < eps])
            cond = cond/ (cond.sum(axis=1)[:,None])
            loss = (cond-pred)**2
            loss = torch.mean(loss)
            if torch.any(~torch.isfinite(loss)):
                print('bad LOSS MSE')
#                 print('COND mean',cond_mean[i])
#                 print('Cond var',cond_var[i])
                pdb.set_trace()
                breakpoint()
           
        elif loss_function == 'hellinger':
            cond[cond < eps] += (eps - cond[cond < eps])
            pred[pred < eps] += (eps - pred[pred < eps])
            cond = cond/ (cond.sum(axis=1)[:,None])
            loss = .5 * torch.sum( (torch.sqrt(cond) - torch.sqrt(pred))**2, axis = 1 )
            loss = torch.mean(loss)
            if torch.any(~torch.isfinite(loss)):
                print('bad LOSS HELLINGER')
#                 print('COND mean',cond_mean[i])
#                 print('Cond var',cond_var[i])
                pdb.set_trace()
                breakpoint()
        
        elif loss_function == "joint_mse":
            cond[cond < eps] += (eps - cond[cond < eps])
            n_vals = params[:,3]
            b,beta,gamma = 10**params[:,0],10**params[:,1],10**params[:,2]
            mu_n = b/beta
            var_n = (mu_n)*(b + 1)
            n_binom = mu_n**2/(var_n - mu_n)
            p_binom = mu_n/var_n
            
            
            # nascent marginal probability
            comb =  torch.lgamma(n_vals + n_binom) - torch.lgamma(n_binom) + torch.lgamma(n_vals + 1) 
            log_p_nascent = comb + p_binom*torch.log(p_binom) + n_vals*torch.log(1-p_binom)
            
            # joint probability
            pred_joint = torch.exp(log_p_nascent[:,None])*pred
            pred_joint[pred_joint < eps] += (eps - pred_joint[pred_joint < eps])
            
            loss = (cond-pred_joint)**2
            loss = torch.sum(loss)
            
        elif loss_function == "joint_hellinger":
            cond[cond < eps] += (eps - cond[cond < eps])
            n_vals = params[:,3]
#             print('n_vals', n_vals[0:10])
            b,beta,gamma = 10**params[:,0],10**params[:,1],10**params[:,2]
            mu_n = b/beta
#             print('mu_n', mu_n[0:10])
            var_n = (mu_n)*(b + 1)
#             print('var_n', var_n[0:10])
            n_binom = mu_n**2/(var_n - mu_n)
#             print('n_binom',n_binom[0:10])
            p_binom = mu_n/var_n
#             print('p_binom',p_binom[0:10])
            
            
            # nascent marginal probability
            comb =  torch.lgamma(n_vals + n_binom) - torch.lgamma(n_binom) - torch.lgamma(n_vals + 1) 
            log_p_nascent = comb + n_binom*torch.log(p_binom) + n_vals*torch.log(1-p_binom)
            
            # joint probability
            pred_joint = torch.exp(log_p_nascent[:,None])*pred
            pred_joint[pred_joint < eps] += (eps - pred_joint[pred_joint < eps])


#             print('log_p_nascent',log_p_nascent[0:10]) 
#             print('p_nascent',torch.exp(log_p_nascent[0:10]))

#             print('PRED',pred[0,0:10])

#             print('PRED JOINT',pred_joint[0,0:10])
#             print('COND',cond[0,0:10])


            loss = .5 * torch.sum( (torch.sqrt(cond) - torch.sqrt(pred_joint))**2, axis = 1 )
            loss = torch.sum(loss)
        return(loss)
        
    def save(self, save_path):
        # save model
        torch.save(self.model, save_path + '_MODEL')
        
        
    def get_cond_P(self,param,m_limit):
        ''' Return conditional P given params.
        '''
        
        self.model.eval()
        weights, cond_means, cond_stds = self.model.forward(param)


        m = torch.arange(m_limit)
        P = self.eval_cond_P(m,weights,cond_means,cond_stds)
        
        return(P)
