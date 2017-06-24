# coding=utf-8
from __future__ import division
'''
The MIT License (MIT)

Copyright 2017 Siemens AG, University of Cambridge

Authors: Stefan Depeweg, José Miguel Hernández-Lobato

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np
import sys
sys.path.append('../')

from models.AD import AD
import time

import os
import pdb
import pylab as plt
import pickle
import math
import theano
import theano.tensor as T
import lasagne

seed = 3
np.random.seed(seed)

params_task = {}
params_task['task'] = 'IndustrialBenchmark'
params_task['seed'] = seed
params_task['history'] = 15 # time-embedding


dire = '../environment/out/' 
X_train = np.loadtxt(dire+'X_train.txt')
Y_train = np.loadtxt(dire+'Y_train.txt')[:,None]

# data is (Setpoint,A(t-14)..A(t),R(t-15)..R(t-1)) -> R(t)

# We need starting states: Setpoint,A(t-14)..A(t),R(t-14)..R(t)

X_start = X_train[:,1:].reshape((X_train.shape[0],params_task['history'],4))
X_start[:,:,-1] = np.hstack((X_start[:,1:,-1],Y_train))

X_start = np.hstack((X_train[:,:1],X_start.reshape((X_start.shape[0],X_start.shape[1]*X_start.shape[2]))))

params_task['state_dim'] = X_start.shape[1]
params_task['action_dim'] = 3


dire = 'models/'
alpha = sys.argv[1]
fstr = 'AD_'+alpha+ '.p'
params_model = pickle.load(open(dire+fstr,'rb'))['params_model']
params_model['saved'] = dire+fstr

# we transform the reward using a logit-log transformation
X1 = X_start[:,1:].reshape((X_start.shape[0],params_task['history'],4))
At = X1[:,:,:3]
Rt = X1[:,:,3:4]

[a_x,a_y,b_x,b_y] = params_model['bounds']
Rt = np.log(1+Rt)

from scipy.special import logit

Rt = logit((Rt - a_x) / (b_x- a_x))

X1[:,:,3:4] = Rt
X_start = np.hstack((X_start[:,0:1],X1.reshape((X1.shape[0],params_task['history']*4))))

# load the model
print params_model['saved']
model = AD(params_model,params_task,X_start,X_start[:,-1:])
model.loadWeights() 


# define the policy search parameters
params_controller = {}
params_controller['saved'] =  False
params_controller['learning_rate'] = 0.0001
params_controller['name'] = 'controller'
params_controller['T'] = 75
params_controller['epochs'] = 750
params_controller['batchsize'] = 25
params_controller['samples'] = 25


def policy(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, X_start.shape[1]),input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=20,nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=20,nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_hid2,num_units=3,nonlinearity=lasagne.nonlinearities.tanh)
    return l_out


import controller.PolicySearch
contrl = controller.PolicySearch.PolicySearch(params_controller,params_task,X_start,model,policy())

errs = []
trace = []

from timeit import default_timer as timer
start = timer()
time_epochs = []

model.bb_alpha.network.update_randomness(params_controller['samples'])
for j in range(params_controller['epochs']):
    errs = []
    inds = np.random.permutation(X_start.shape[0])
    for k in range(100):
        ind = inds[k*params_controller['batchsize']:(k+1)*params_controller['batchsize']]
        model.bb_alpha.network.update_randomness(params_controller['samples'])
        e = contrl.train_func(X_start[ind])
        errs.append(e)

    end = timer()
    time_e = end-start
    time_epochs.append(time_e)
    atime_e = np.mean(time_epochs[-5:])
    rest_time = int(atime_e * (params_controller['epochs'] - (j+1)))
    rest_hours,rest_seconds = divmod(rest_time,60*60)
    rest_minutes,_ = divmod(rest_seconds,60)
    err = np.mean(errs)
    print 'Remaining: ' + str(rest_hours) + 'h:'  + str(rest_minutes) +  'm,  Policy Cost: ' + str(err) 
    trace.append(err)
    start = timer()


weights = []
for p in lasagne.layers.get_all_params(contrl.policy,trainable=True):
    weights.append(p.get_value())

saved = {}
saved['params_model'] = params_model
saved['params_controller'] = params_controller
saved['params_task'] = params_task
saved['model_norm'] = [model.mean_X,model.std_X,model.mean_Y,model.std_Y]
saved['trace'] = trace
saved['controller_weights'] = weights
dire = 'controller/'
pickle.dump(saved,open(dire + fstr,'wb'))
