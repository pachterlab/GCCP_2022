import numpy as np
import time

import ypred_module as ypm
import exact_cme as cme
import train_conditional as train
import tools_conditional as tools
import direct_module as direct


# Figure 2a: Time vs. Hellinger distance
data_set = train.load_data(1,'./data/','256_test_full')
parameters = data_set[:,0]

# define functions to define how many std the grid will extend
xmax_fun_1std = lambda x : x/4.0
xmax_fun_4std = lambda x : x
xmax_fun_10std = lambda x : x*(10./4.)
xmax_fun_20std = lambda x : x*5

compare_array_norm_hellinger = np.zeros((6,256,2))
compare_array_hellinger = np.zeros((6,256,2))

for i in range(256):
    print(i)
    # quadvec 20 sigma approximation
    t1 = time.time()
    y_20std = cme.calculate_exact_cme(parameters[i],method='quad_vec',xmax_fun=xmax_fun_20std)[1]
    t2 = time.time()
    
    
    ss = y_20std.shape[0]*y_20std.shape[1]
    compare_array_norm_hellinger[0,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[0,i,1] = hellinger(y_20std,y_20std)/(ss)
    compare_array_hellinger[0,i,0] = (t2-t1)
    compare_array_hellinger[0,i,1] = hellinger(y_20std,y_20std)
    
    
    # quadvec 10 sigma approximation
    t1 = time.time()
    y_10std = cme.calculate_exact_cme(parameters[i],method='quad_vec',xmax_fun=xmax_fun_10std)[1]
    t2 = time.time()
    
    
    compare_array_norm_hellinger[1,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[1,i,1] = hellinger(y_20std,y_10std)/(ss)
    compare_array_hellinger[1,i,0] = (t2-t1)
    compare_array_hellinger[1,i,1] = hellinger(y_10std,y_10std)
    
    
    # quadvec 4 sigma approximation
    t1 = time.time()
    y_4std = cme.calculate_exact_cme(parameters[i],method='quad_vec',xmax_fun=xmax_fun_4std)[1]
    t2 = time.time()
    
    
    compare_array_norm_hellinger[2,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[2,i,1] = hellinger(y_20std,y_4std)/(ss)
    compare_array_hellinger[2,i,0] = (t2-t1)
    compare_array_hellinger[2,i,1] = hellinger(y_4std,y_4std)
    
    # fixed quad approximation
    t1 = time.time()
    y_fixed_quad = cme.calculate_exact_cme(parameters[i],method='fixed_quad',xmax_fun=xmax_fun_4std)[1]
    t2 = time.time()
    
    
    compare_array_norm_hellinger[3,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[3,i,1] = hellinger(y_20std,y_fixed_quad)/(ss)
    compare_array_hellinger[3,i,0] = (t2-t1)
    compare_array_hellinger[3,i,1] = hellinger(y_20std,y_fixed_quad)
    
    # NN sigma approximation
    nas_range = np.arange(y_20std.shape[0])
    mat_range = np.arange(y_20std.shape[1])

    t1 = time.time()
    y_NN = ypm.get_prob(parameters[i],nas_range,mat_range)
    t2 = time.time()
    
    
    compare_array_norm_hellinger[4,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[4,i,1] = hellinger(y_20std,y_NN)/(ss)
    compare_array_hellinger[4,i,0] = (t2-t1)
    compare_array_hellinger[4,i,1] = hellinger(y_20std,y_NN)
    
    #NB approximation
    t1 = time.time()
    N,M = np.meshgrid(nas_range,mat_range,indexing='ij')
    y_NB = ypm.approximate_conditional_tensorval(parameters[i],N,M).detach().numpy()
    t2 = time.time()
    compare_array_norm_hellinger[5,i,0] = (t2-t1)/(ss)
    compare_array_norm_hellinger[5,i,1] = hellinger(y_20std,y_NB)/(ss)
    compare_array_hellinger[5,i,0] = (t2-t1)
    compare_array_hellinger[5,i,1] = hellinger(y_20std,y_NB)



np.save('../results/compare_array_norm_hellinger',compare_array_norm_hellinger)
np.save('../results/compare_array_hellinger',compare_array_hellinger)



# now, timing and hellinger distance for direct approximation
hellinger_direct = np.zeros(256)
times_direct = np.zeros(256)
hellinger_direct_unnorm = np.zeros(256)

for i in range(256):
    rate_vec_1 = data_set[i][0]
    pdf_1 = data_set[i][1]
    ss = pdf_1.shape[0]*pdf_1.shape[1]
    t1 = time.time()
    pred = direct.predict_pmf(rate_vec_1,pdf_1.shape[0],pdf_1.shape[1])
    t2 = time.time()
    
    times_direct[i] = (t2-t1)/ss
    
    hellinger_direct[i] = hellinger(pdf_1,pred)/ss
    hellinger_direct_unnorm[i] = hellinger(pdf_1,pred)

np.save('../results/direct_model256_3_4t_hellinger_norm',hellinger_direct)
np.save('../results/direct_model256_3_4t_times_norm',times_direct)


# Figure 2b. Hellinger Distances 
data_array = train.load_data(3,'./data/','256_test_full')

NN_hellinger = np.zeros(256*3)
rand_hellinger = np.zeros(256*3)
NB_hellinger = np.zeros(256*3)
DR_hellinger = np.zeros(256*3)

for i in range(256*3):

    pdf = data_array[i][1]
    p_in = data_array[i][0]

    nas_range = np.arange(pdf.shape[0])
    mat_range = np.arange(pdf.shape[1])

    predicted = ypm.get_prob(p_in,nas_range,mat_range)
    predicted_rand = ypm.get_prob(p_in,nas_range,mat_range,rand_weights=True)
    predicted_DR = direct.predict_pmf(p_in,pdf.shape[0],pdf.shape[1])
    DR_hellinger[i] = tools.hellinger(predicted_DR,pdf)
    NN_hellinger[i] = tools.hellinger(predicted,pdf)
    rand_hellinger[i] =  tools.hellinger(predicted_rand,pdf)
    
    N,M = np.meshgrid(range(pdf.shape[0]),range(pdf.shape[1]),indexing='ij')
    NB = ypm.approximate_conditional_tensorval(p_in,N,M).detach().numpy()
    NB_hellinger[i] = tools.hellinger(pdf,NB)


np.save('./results/NN_hellinger',NN_hellinger)
np.save('./results/NB_hellinger',NB_hellinger)
np.save('./results/rand_hellinger',rand_hellinger)
np.save('./results/direct_model256_3_4t_hellinger_unnorm',DR_hellinger)


# Figure 2c. TIMING for ONE point
NN_times_onepoint = np.zeros(256*3)
NB_times_onepoint = np.zeros(256*3)
direct_times_onepoint = np.zeros(256*3)

for i in range(256*3):
    rate_vec = data_array[i][0]

    t1 = time.time()
    pred = ypm.get_prob(rate_vec,n_range=np.arange(0),m_range=np.arange(0))
    t2 = time.time()

    times_NN_onepoint[i] = (t2-t1)

    t1 = time.time()
    pred = direct.predict_point(rate_vec,0,0)
    t2 = time.time()
    
    t1 = time.time()
    N,M = np.meshgrid(np.arange(0),np.arange(0),indexing='ij')
    y_NB = ypm.approximate_conditional_tensorval(rate_vec,N,M).detach().numpy()
    t2 = time.time()

    times_NB_onepoint[i] = (t2-t1)

np.save('./results/NN_times_onepoint',NN_times_onepoint)
np.save('./results/NB_times_onepoint',NB_times_onepoint)
np.save('./results/direct_model256_3_4t_times_onepoint',direct_times_onepoint)








