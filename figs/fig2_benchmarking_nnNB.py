import numpy as np
import time


import sys
sys.path.append('../')


import ypred_module as ypm
import exact_cme as cme
import train_conditional as train
import tools_conditional as tools
import nnNB_module as nnNB_module_new


model = nnNB_module_new.model1



# Figure 2a: Time vs. Hellinger distance
data_set = train.load_data(1,'./data/','256_test_full')

parameters = [i[0] for i in data_set]

def get_KLD(true_prob,approx):
    eps = 1e-10
    true_prob = true_prob/true_prob.sum()
    approx[approx<eps] = eps

    approx = approx/approx.sum()
    
    KL = -np.sum(true_prob*np.log(approx/true_prob))
    
    return(KL)

def hellinger(a,b):
    
    return np.sum((np.sqrt(a)-np.sqrt(b))**2)/2
    

def xfun(x):
    return(5*x)


# the dimensions are:
# MMNB, nnNB_old, nnNB_new/1NB by test distributions by time/value
compare_array_norm_hellinger = np.zeros((4,256,2))
compare_array_hellinger = np.zeros((4,256,2))
compare_array_KLD = np.zeros((4,256))


for i in range(256):
    print(i)
    # quadvec 20 sigma approximation
    t1 = time.time()
    y_20std = cme.calculate_exact_cme(parameters[i],method='quad_vec',xmax_fun=xfun)[1]
    t2 = time.time()
    
    
    ss = y_20std.shape[0]*y_20std.shape[1]
    compare_array_norm_hellinger[0,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[0,i,1] = hellinger(y_20std,y_20std)/(ss)
    compare_array_hellinger[0,i,0] = (t2-t1)
    compare_array_hellinger[0,i,1] = hellinger(y_20std,y_20std)
    
    # MMNB approximation
    nas_range = np.arange(y_20std.shape[0])
    mat_range = np.arange(y_20std.shape[1])

    t1 = time.time()
    N,M = np.meshgrid(nas_range,mat_range,indexing='ij')
    y_NB = ypm.approximate_conditional_tensorval(parameters[i],N,M).detach().numpy()
    t2 = time.time()
    compare_array_norm_hellinger[1,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[1,i,1] = hellinger(y_20std,y_NB)/(ss)
    compare_array_hellinger[1,i,0] = (t2-t1)
    compare_array_hellinger[1,i,1] = hellinger(y_20std,y_NB)
    compare_array_KLD[1,i] = get_KLD(y_20std,y_NB)

    # nnNB OLD approximation
    t1 = time.time()
    y_NN = ypm.get_prob(parameters[i],nas_range,mat_range)
    t2 = time.time()
    
    
    compare_array_norm_hellinger[2,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[2,i,1] = hellinger(y_20std,y_NN)/(ss)
    compare_array_hellinger[2,i,0] = (t2-t1)
    compare_array_hellinger[2,i,1] = hellinger(y_20std,y_NN)
    compare_array_KLD[2,i] = get_KLD(y_20std,y_NN)
    
    
    # nnNB NEW approximation 
    t1 = time.time()
    y_NN_new = nnNB_module_new.nnNB_prob(parameters[i],
                            n = nas_range,m = mat_range,model = model,use_old=False)
    t2 = time.time()
    
    
    compare_array_norm_hellinger[3,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[3,i,1] = hellinger(y_20std,y_NN_new)/(ss)
    compare_array_hellinger[3,i,0] = (t2-t1)
    compare_array_hellinger[3,i,1] = hellinger(y_20std,y_NN_new)
    compare_array_KLD[3,i] = get_KLD(y_20std,y_NN_new)




np.save('./results/compare_array_norm_hellinger_1NB',compare_array_norm_hellinger)
np.save('./results/compare_array_hellinger_1NB',compare_array_hellinger)
np.save('./results/compare_array_KLD_1NB',compare_array_KLD)


# Figure 2b. Hellinger Distances 
data_array = train.load_data(3,'./data/','256_test_full')
NN_new_hellinger = np.zeros(256*3)


for i in range(256*3):

    pdf = data_array[i][1]
    p_in = data_array[i][0]

    nas_range = np.arange(pdf.shape[0])
    mat_range = np.arange(pdf.shape[1])
    
    predicted_NN_new = nnNB_module_new.nnNB_prob(p_in,
                            n = nas_range,m = mat_range,model = model,use_old=False)
    
    NN_new_hellinger[i] = tools.hellinger(predicted_NN_new,pdf)


np.save('./results/NN_hellinger_1NB',NN_new_hellinger)



# Figure 2c. TIMING for ONE point
NN_new_times_onepoint = np.zeros(256*3)

for i in range(256*3):
    rate_vec = data_array[i][0]

    t1 = time.time()
    pred = nnNB_module_new.nnNB_prob(p_in,
                            n = np.arange(1),m = np.arange(1),model = model,use_old=False)
    t2 = time.time()

    NN_new_times_onepoint[i] = (t2-t1)


np.save('./results/NN_times_onepoint_1NB',NN_new_times_onepoint)







    
    

