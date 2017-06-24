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
import copy

seed = 3
np.random.seed(seed)

dire = '../environment/out/' 

X_train = np.loadtxt(dire+'X_train.txt')
Y_train = np.loadtxt(dire+'Y_train.txt')[:,None]
X_test = np.loadtxt(dire+'X_test.txt')
Y_test = np.loadtxt(dire+'Y_test.txt')[:,None]

state_dim = X_train.shape[1]
tar_dim = Y_train.shape[1]

params_task = {} 
params_task['seed'] = seed
params_task['history'] = 15 # time-embedding
params_task['state_dim'] = X_train.shape[1]
params_task['r_dim'] = Y_train.shape[1]


params_model = {}
params_model['mode'] = 'AD'
params_model['saved'] = False
params_model['epochs'] = 3000
params_model['batchsize'] = int(X_train.shape[0]/25)
params_model['alpha'] = float(sys.argv[1])
params_model['learn_rate'] = np.float32(0.001)
params_model['samples'] =  50
params_model['dimh'] = 75
params_model['graph'] = [params_task['state_dim'] ,params_model['dimh'],params_model['dimh'],1]

X = X_train
Y = Y_train

# we transform the reward using a logit-log transformation
X1 = X[:,1:].reshape((X.shape[0],params_task['history'],4))
At = X1[:,:,:3]
Rt = X1[:,:,3:4]

a_x = np.log(1+Rt.min(axis=0)*0.95)
a_y = np.log(1+Y.min(axis=0)*0.95)
b_x = np.log(1+Rt.max(axis=0)*1.05)
b_y = np.log(1+Y.max(axis=0)*1.05)

Rt = np.log(1+Rt)
Y = np.log(1+Y)

from scipy.special import logit

Rt = logit((Rt - a_x) / (b_x- a_x))
Y =  logit((Y -  a_y) / (b_y - a_y))

X1[:,:,3:4] = Rt
X = np.hstack((X[:,0:1],X1.reshape((X1.shape[0],params_task['history']*4))))


params_model['bounds'] = [a_x,a_y,b_x,b_y]

# predict \Delta_t
Y[:,0] = Y[:,0] - X[:,-1]

model = AD(params_model,params_task,X,Y)
model.train() 

saved = {}
saved['params_model'] = params_model
saved['params_task'] = params_task
saved['model_norm'] = [model.mean_X,model.std_X,model.mean_Y,model.std_Y]
saved['model_weights'] = model.get_weights() 

dire = 'models/'
outstr = params_model['mode'] + '_' + str(params_model['alpha'])  + '.p'
pickle.dump(saved,open(dire + outstr,'wb'))

