import ypred_module as ypm
import train_conditional as train
import tools_conditional as tools

import numpy as np
import time


data_path = './data/'
training_set = train.load_data(number_of_files = 6 ,file_path = data_path ,name = '256_train')
validation_set = train.load_data(number_of_files = 2 ,file_path = data_path ,name = '256_valid')


model_config = {
    'input_dim' : 7,
    'npdf' : 10,
    'h1_dim' : 256,
    'h2_dim' : 256,
    }

train_config = {
    'num_epochs' : 35,
    'lr' : 1e-3,
    'weight_decay' : 0,
    'batchsize' : 100,
    'metric' : 'kld'
    }
    
    t1 = time.time()

    model, train_metvals, valid_metvals = train.train(training_set_,validation_set,model_config,train_config)

    t2 = time.time()
    t = t2-t1
    
    timing[1,i] = t


#tools.save_model_and_meta(model,model_config,train_config,train_loss,valid_loss,time,path,name)