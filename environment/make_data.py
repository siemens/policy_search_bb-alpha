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

import os
import pdb
import pylab as plt
import pickle
import math
import itertools
from scipy.stats import rankdata
import theano
import theano.tensor as T

import copy
## Free params ##

history = 15

'''
generate Batch
'''
from industrialbenchmark.IDS import IDS 

setpoints = range(10,110,10)

n_runs = 10 # episodes per setpoint
n_runs_train = 7
n_runs_test = 3
steps = 1000  # steps per episodes
# model:  R(t) = f(A(t-14),..,A(t),R(t-15),R(t-1),Setpoint)
# action_dim = 3, R_dim = 1

X_train = np.zeros((len(setpoints)*n_runs_train*(steps-(history+5)),15*(3+1) + 1))
Y_train = np.zeros((len(setpoints)*n_runs_train*(steps-(history+5)),1))
X_test = np.zeros((len(setpoints)*n_runs_test*(steps-(history+5)),15*(3+1) + 1))
Y_test = np.zeros((len(setpoints)*n_runs_test*(steps-(history+5)),1))

idx_tr = 0
idx_te = 0
for k in range(n_runs):
    for s in setpoints:
        env = IDS(s)
        states = np.zeros((steps,env.visibleState().shape[0]))
        for j in range(steps):
            env.step(2*np.random.rand(3)-1)
            states[j,:] = env.visibleState()

            if j >= history + 5:
                feat = np.hstack((states[j-(history-1):j+1,1:4],states[j-history:j,-2:-1]))
                feat = np.hstack((s,feat.ravel()))

                if k < n_runs_train:
                    X_train[idx_tr] = feat
                    Y_train[idx_tr] = states[j,-2:-1]
                    idx_tr += 1
                else:
                    X_test[idx_te] = feat
                    Y_test[idx_te] = states[j,-2:-1]
                    idx_te += 1


dire =  'out/'
np.savetxt(dire + 'X_train.txt',X_train,fmt='%5.4f')
np.savetxt(dire + 'Y_train.txt',Y_train,fmt='%5.4f')
np.savetxt(dire + 'X_test.txt',X_test,fmt='%5.4f')
np.savetxt(dire + 'Y_test.txt',Y_test,fmt='%5.4f')
