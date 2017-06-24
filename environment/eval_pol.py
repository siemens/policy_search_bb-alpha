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
#np.random.seed(1)
import sys
from scipy.special import logit
import cPickle as pickle
import lasagne
import theano
import theano.tensor as T
from industrialbenchmark.IDS import IDS


def policy(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 61),input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=20,nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=20,nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_hid2,num_units=3,nonlinearity=lasagne.nonlinearities.tanh)
    return l_out

pol = policy()
x = T.matrix()
policy_network = theano.function(inputs=[x],outputs=lasagne.layers.get_output(pol,x))


def predict(st,norm,bounds):
    
    rew = np.log(1+ (st[:,-1:]))

    a_x = bounds[0]
    b_x = bounds[2]

    eps = 1e-5

    rew = np.clip(rew,a_x+eps,b_x-eps)

    rew = logit((rew - a_x) / (b_x - a_x))

    st [:,-1:] = rew  

    State  = np.zeros((1,61))
    State[0,:] = np.hstack((st[0,0],st[:,[1,2,3,-1]].ravel()))

    X = (State  - norm[0]) / norm[1]
    return np.round(policy_network(X)[0,:],4)




def evaluate(fstr,do_warmup=True):
    co = pickle.load(open(fstr,'rb'))
    np.random.seed(1)
    bounds = co['params_model']['bounds']
    norm =  co['model_norm']
    
    for j,p in enumerate(lasagne.layers.get_all_params(pol,trainable=True)):
        p.set_value(co['controller_weights'][j])

    n_evals  = 25
    data = np.zeros((10,n_evals,100))

    for k in range(10):
        for j in range(n_evals):
            env = IDS((k+1)*10)
            if do_warmup == True:
                burnin = 15  + np.random.randint(50)
                st = np.zeros((15,env.visibleState().shape[0]))
                for l in range(burnin):
                    at = 2*np.random.rand(3) -1
                    env.step(at)
                    st = np.vstack((st[1:,:],env.visibleState().reshape((1,7))))
            else:
                st = env.visibleState().reshape((1,-1))
                st = np.tile(st,[15,1])
    
            traj = np.zeros(100)
            for l in range(100):
                at = predict(np.copy(st),norm,bounds)
                env.step(at)
                st = np.vstack((st[1:,:],env.visibleState().reshape((1,7))))
                traj[l] = env.visibleState()[-1]
            data[k,j,:] = traj
    print fstr + ': ' + str(data.mean())
    return data
        
        
