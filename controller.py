#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:22:59 2017

@author: ghulamahmedansari
"""


from memory import Memory
import numpy as np

class Control(object):
    def __init__(self, config):
        self.config = config
        self.qtable = Memory(config)

    def update_table(self,value_list):
        for s,a,r in value_list:
            self.qtable.add(s,a,r)

    def getaction(self,state):
        vals = np.array([self.qtable.getQval(state,action) for action in range(self.config.action_set_size)])
        return np.random.choice(np.flatnonzero(vals == vals.max()))
