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
#setting up seed for repititin experiments
random.seed(1)
np.random.seed(10)

class Memory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.Qtable = [pylru.lrucache(config.memory_size) for action in range(self.config.action_set_size)]
        self.config = config
        self.flann = pyflann.FLANN()
        self.count = [0]*self.config.action_set_size

    def add(self, state, action, Return):
        key = tuple(state)
        if key not in self.Qtable[action]:
            self.Qtable[action][key] = Return
        else:
            self.Qtable[action][key] = max(self.Qtable[action][key],Return)

        self.count[action] += 1

    def getQval(self,state,action):
        key = tuple(state)
        if key in self.Qtable[action]:
            return self.Qtable[action][key]
        else:
            if self.count[action] > self.config.k: #to handle borderline cases
                dataset = np.array([i for i in self.Qtable[action].keys()])
                nearest_k_indices, _ = self.flann.nn(dataset, state, self.config.k, algorithm="kmeans", branching=32, iterations=7, checks=16)
                nearest_k_indices = nearest_k_indices[0]
                self.Qtable[action][key] = np.mean([self.Qtable[action][dataset[i]] for i in nearest_k_indices])
            else:
                self.Qtable[action][key] = 0
        return self.Qtable[action][key]
