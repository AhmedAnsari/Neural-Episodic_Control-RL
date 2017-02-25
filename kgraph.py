#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:36:32 2017

@author: ghulamahmedansari
"""

import pykgraph
import pykgraph
import numpy as np

class Kgraph(object):
    def __init__(self,config):
        self.dataset = []
        self.config = config
        
    def add_item(self,i,item):
        self.dataset.append(item)
        
    def build(self,ntrees):
        np_dataset = np.array(self.dataset, dtype=np.float32) #kgraph only supports float32 and float 64
        self.index = pykgraph.KGraph(np_dataset, 'euclidean') #it is not optimized for float64
        self.index.build(reverse=-1, K=self.config.k)
        self.ntrees = ntrees
        
    def get_nns_by_vector(self,vector,k):
#        from numpy import random
#        np_vector = random.rand(1,64) 
#        print np_vector, np_vector.shape, type(np_vector[0][0])
        
        np_vector = np.asarray(vector, dtype=np.float32)
        np_vector = np_vector[np.newaxis,:]/1
#        print np_vector, np_vector.shape, type(np_vector[0][0])
        return self.index.search(np_vector, k, P=100)[0]