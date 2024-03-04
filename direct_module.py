import os,sys,importlib_resources
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as plt

# +
# if no model is given, will use the direct final trained model stored in ./models/
package_resources = importlib_resources.files("direct_module")
model_path = os.path.join(package_resources,'models/direct_models/direct_256u_3l_4t_MODEL')


# -




class MLP(nn.Module):

    def __init__(self, input_size, num_hidden_units, num_hidden_layers, output_size,activate='relu'):
    	super().__init__()
    	self.activate = activate
    	self.module_list = nn.ModuleList([])
    	self.module_list.append(nn.Linear(input_size,num_hidden_units))


    	for k in range(num_hidden_layers-1):
    		self.module_list.append(nn.Linear(num_hidden_units, num_hidden_units))


    	self.module_list.append(nn.Linear(num_hidden_units,output_size))




    def forward(self, x):

    	for f in self.module_list[:-1]:

    		x = f(x)

    		if self.activate == 'relu':
    			x = F.relu(x)
    		elif self.activate == 'sigmoid':
    			x = F.sigmoid(x)

    	x = self.module_list[-1](x)

    	return x


def train_MLP(model_config, train_config, train_set, valid_set):

	input_size = model_config['input_size']
	output_size = model_config['output_size']
	num_hidden_units = model_config['num_hidden_units']
	num_hidden_layers = model_config['num_hidden_layers']
	activate = model_config['activate']


	lr = train_config['lr']
	num_epochs = train_config['num_epochs']
	batchsize = train_config['batchsize']
	weight_decay = train_config['weight_decay']
	learn_log = train_config['learn_log']


	trials = int(len(train_set)/batchsize)

	model = MLP(input_size, num_hidden_units, num_hidden_layers, output_size, activate)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay) 

	valid_loss = np.zeros(num_epochs)
	train_loss = np.zeros(num_epochs)

	for e in range(num_epochs):
		print('Epoch: ',e+1)

		# set up model 
		model.train()

		# shuffle data (indeces)
		idx = torch.randperm(train_set.shape[0])
		t_shuffled = train_set[idx]
		
		# set up array to store losses
		trial_loss = np.zeros(trials)

		for i in range(trials):

			trial_set = t_shuffled[batchsize*i:batchsize*i+batchsize]

			if learn_log == 'True':
				true_prob = torch.log(trial_set[:,-1].view(-1,1))
			else:
				true_prob = trial_set[:,-1].view(-1,1)

			rate_vec = trial_set[:,0:-1]

			optimizer.zero_grad()

			pred_prob = model(rate_vec)

			if learn_log == 'True':
				loss = F.mse_loss(pred_prob,true_prob,reduction='mean')
			else:
				loss = F.mse_loss(pred_prob,true_prob,reduction='sum')


			trial_loss[i] = loss.item()

			loss.backward()

			optimizer.step()



		train_loss[e] = np.mean(trial_loss)

		# check on validation data
		model.eval()
		valid_true_prob = torch.log(valid_set[:,-1].view(-1,1))
		valid_rate_vec = valid_set[:,0:-1]
		valid_pred_prob = model(valid_rate_vec)
		v_loss = F.mse_loss(valid_pred_prob,valid_true_prob)
		valid_loss[e] = torch.mean(v_loss)


	return train_loss, valid_loss, model



class TrainedModel():

    def __init__(self, path, input_size, num_hidden_units, num_hidden_layers, output_size):
        
        meta = np.load(path + '_meta.npy',allow_pickle=True)
        
        self.meta = meta
        self.train_loss = meta[0]
        self.valid_loss = meta[1]
        self.model_config = meta[2]
        self.train_config = meta[3]
        
        input_size = self.model_config['input_size']
        output_size = self.model_config['output_size']
        num_hidden_layers = self.model_config['num_hidden_layers']
        num_hidden_units = self.model_config['num_hidden_units']

        
        self.model = mdl.MLP(input_size, num_hidden_units, num_hidden_layers, output_size)
        self.model.load_state_dict(torch.load(path+'_MODEL'))
        self.model.eval()

        




# load in model
model_direct = MLP(input_size=5, num_hidden_units = 256, num_hidden_layers = 3, output_size = 1)
model_direct.load_state_dict(torch.load(model_path))
model_direct.eval()


def predict_pmf(rate_vec,n_len,m_len,model=model_direct):
    
    a = np.ones((n_len,m_len))
    b = np.arange(n_len).reshape(-1,1)
    n = np.multiply(b,a).flatten()
    
    rate_vecs = torch.tensor([rate_vec[0],rate_vec[1],rate_vec[2],0,0],dtype=torch.float).repeat(m_len*n_len,1)
    
    rate_vecs[:,-1] =  torch.arange(m_len).repeat(n_len)
    rate_vecs[:,-2] =  torch.tensor(n)
    
    logP = model(rate_vecs).detach().numpy().flatten()
    predicted_pdf = np.exp(logP).reshape((n_len,m_len))
            
    return predicted_pdf


def predict_point(rate_vec,n,m,model=model_direct):
    
    vec = torch.tensor([rate_vec[0],rate_vec[1],rate_vec[2],n,m],dtype=torch.float)

    logP = model(vec).detach().numpy()
    
    return np.exp(logP)



