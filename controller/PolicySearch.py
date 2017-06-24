# coding=utf-8
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
import cPickle as pickle
import theano
import theano.tensor as T
from  theano import shared
import pdb
from collections import OrderedDict

import lasagne
from lasagne.updates import adam

class PolicySearch:
    def __init__(self,params,params_task,X,model,policy):
        
        self.rng = np.random.RandomState()

        self.model = model
        self.policy = policy

        self.params = params
        self.params_task = params_task


        self.x = T.matrix('x')
        cost  =  self.control(self.x)

        self.fwpass  = theano.function(inputs=[self.x], outputs = cost,allow_input_downcast=True)
        self.train_func = theano.function(inputs=[self.x],outputs=[cost], updates=adam(cost,lasagne.layers.get_all_params(self.policy,trainable=True),learning_rate=self.params['learning_rate']))

        self.policy_network = theano.function(inputs=[self.x],outputs=self.predict(self.x))


    def control(self,st):
        srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        # do n roll-outs for each starting state
        n = self.params['samples']
        st_s = T.tile(st,[n,1])

        onoise =  srng.normal(size=(st_s.shape[0],1,self.params['T']))
        inoise = T.sqrt(st.shape[1]) * srng.normal(size=(n,st.shape[0],self.params['T']))

        ([_,_,R], updates) = theano.scan(fn=self._step,outputs_info=[st_s,T.as_tensor_variable(0),None],n_steps=self.params['T'],non_sequences=[onoise,inoise])
        return R.mean()

    def _step(self,st_s,t,onoise,inoise):
            
        on_t =  onoise[:,:,t]
        in_t = inoise[:,:,t:t+1]
        
        # get action
        at_s = self.predict(st_s)

        # obtain new steering variables
        A_t1 = self.aAction(st_s,at_s)
        
        
        # time-shift steerings 1 into the future
        # (A(t-15),..A(t)  ->  A(t-14),..,A(t+1)
        st_s3 = st_s[:,1:].reshape((st_s.shape[0],self.params_task['history'],4))
        st1_s3 = T.set_subtensor(st_s3[:,:,:3],T.concatenate((st_s3[:,1:,:3],T.shape_padaxis(A_t1,1)),axis=1))
        xt1_s = T.concatenate((st_s[:,:1],st1_s3.reshape((st_s.shape[0],st_s.shape[1]-1))),axis=1)

        
        # Obtain \delta R(t+1) by BNN
        xt1_s  = xt1_s.reshape((self.params['samples'],xt1_s.shape[0]/self.params['samples'],xt1_s.shape[1]))
        drt1_s,vdrt1_s = self.model.predict(xt1_s,mode='symbolic',provide_noise=True,noise=in_t)
        drt1_s = drt1_s.reshape((drt1_s.shape[0]*drt1_s.shape[1],drt1_s.shape[2]))
        vdrt1_s = vdrt1_s.reshape((vdrt1_s.shape[0]*vdrt1_s.shape[1],vdrt1_s.shape[2]))

        # sample from output noise
        drt1_s = on_t * T.sqrt(vdrt1_s) + drt1_s 
        
        #obtain R(t+1) by adding \delta R(t+1)
        rt1_s = st_s[:,-1:] + drt1_s[:,0:1] 


        # undo log-logit transformation to obtain unnormalized reward
        rew1 = 1. / (1. +  T.exp(-rt1_s)) # undo logit
        rew1 = rew1  * (self.model.params['bounds'][3] - self.model.params['bounds'][1])  + self.model.params['bounds'][1]
        rew1 = T.exp(rew1) - 1

        # update time-embedding: R(t-15)..R(t) -> R(t-14) .. R(t+1)
        st1_s3 = T.set_subtensor(st1_s3[:,:,3:],T.concatenate((st1_s3[:,1:,3:],T.shape_padaxis(rt1_s,1)),axis=1))
        st1_s = T.concatenate((st_s[:,:1],st1_s3.reshape((st_s.shape[0],st_s.shape[1]-1))),axis=1)

        return [st1_s,t+1,rew1[:,0]]


    def aAction(self,st,delta):
        '''
        for simplicity we obtain the transformation
        of actions to steerings from the benchmark

        (This is optional, we would simply need to extend our  model to  f(A,R,action \in [-1,1]^3) to learn the following transformation for full generalibility)
        
        '''

        maxRequiredStep = np.sin((15./180)*np.pi)
        gsBound = 1.5
        gsSetPointDependency = 0.02
        gsScale = 2.0*gsBound + 100.*gsSetPointDependency
        shift_step = (maxRequiredStep/0.9)*100./gsScale
        
        A_0 = st[:,-4:-1]

        v_n = A_0[:,0]+1.0*delta[:,0] #
        g_n = A_0[:,1]+10.*delta[:,1]
        s_n = A_0[:,2] + shift_step*delta[:,2]

        # velocity
        A_0 = T.set_subtensor(A_0[:,0],T.clip(v_n ,0.,100.))
        # gain
        A_0 = T.set_subtensor(A_0[:,1],T.clip(g_n ,0.,100.))
        # shift
        A_0 = T.set_subtensor(A_0[:,2],T.clip(s_n   ,0.,100.))

        return A_0


    def predict(self,X):
        X = (X - self.model.mean_X.astype(theano.config.floatX)) / self.model.std_X.astype(theano.config.floatX)
        return lasagne.layers.get_output(self.policy,X)
    
    
    
