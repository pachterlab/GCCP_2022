import ypred_module as ypm
import train_conditional as train
import tools_conditional as tools

import numpy as np
import time

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('-npdf','--npdf',default=10,type=int)

args = parser.parse_args()


npdf = args.npdf


data_path = './data/'
training_set = train.load_data(number_of_files = 2 ,file_path = data_path ,name = '256_train')
validation_set = train.load_data(number_of_files = 2 ,file_path = data_path ,name = '256_valid')

num_params = [1000,10000,20000,30000,40000,50000,60000,70000]
timing = np.zeros((2,len(num_params)))
timing[0,:] = np.array(num_params)

for i,num in enumerate(num_params):
    
    training_set_ = training_set[:num]
    model_config = {
    'input_dim' : 7,
    'npdf' : npdf,
    'h1_dim' : 256,
    'h2_dim' : 256,
    }

    train_config = {
    'num_epochs' : 1,
    'lr' : 1e-3,
    'weight_decay' : 0,
    'batchsize' : 64,
    'metric' : 'kld'
    }
    
    t1 = time.time()

    model, train_metvals, valid_metvals = train.train(training_set_,validation_set[0:1000],model_config,train_config)

    t2 = time.time()
    t = t2-t1
    
    timing[1,i] = t


np.save(f'./results/timing_{npdf}npdf',timing)
