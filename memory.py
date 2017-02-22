#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:42:29 2017

@author: ghulamahmedansari
"""

import random
import numpy as np
import pylru
import pyflann
from annoy import AnnoyIndex
#setting up seed for repititin experiments
random.seed(1)
np.random.seed(10)

class Memory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.config = config
        self.Qtable = [pylru.lrucache(config.memory_size) for action in range(self.config.action_set_size)]
        self.flann = pyflann.FLANN()
        self.count = [0]*self.config.action_set_size

    def add(self, state, action, Return):
        key = tuple(state)
        if key not in self.Qtable[action]:
            self.Qtable[action][key] = Return
            self.count[action] += 1
        else:
            self.Qtable[action][key] = max(self.Qtable[action][key],Return)

    def getQval(self,state,action):
        key = tuple(state)
        if key in self.Qtable[action]:
            return self.Qtable[action][key]
        else:
            if self.count[action] > self.config.k: #to handle borderline cases
                dataset = np.array([i for i in self.Qtable[action].keys()])
                if self.config.KNNmethod == "FLANN":
                    nearest_k_indices, _ = self.flann.nn(dataset, state, self.config.k, algorithm="kmeans", branching=32, iterations=7, checks=16)
                    nearest_k_indices = nearest_k_indices[0]


                elif self.config.KNNmethod == "ANNOY":
                        f = self.config.final_size
                        t = AnnoyIndex(f)
                        count=0
                        for i in self.Qtable[action].keys():
                            t.add_item(count,i)
                            count+=1
                        t.build(10)
                        nearest_k_indices = t.get_nns_by_vector(state,self.config.k)

                self.Qtable[action][key] = np.mean([self.Qtable[action][tuple(dataset[i])] for i in nearest_k_indices])

            else:
                self.Qtable[action][key] = 0
        return self.Qtable[action][key]
