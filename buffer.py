#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 05:25:43 2017

@author: ghulamahmedansari
"""

import numpy as np

class Buffer(object):
    def __init__(self, config):
        self.size = config.T

        self.screens = np.empty((self.size,config.final_size))
        self.rewards = np.empty((self.size))
        self.actions = np.empty((self.size),dtype = np.uint8)
        self.current = 0

    def add(self, screen, action, reward):
        self.screens[self.current] = screen
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.current+=1

    def get_returns(self):
        states = self.screens[0:self.current-1]
        actions = self.actions[1:self.current]
        returns = self.rewards[1:self.current]
        length = returns.shape[0]
        for j in reversed(range(length-1)):
            returns[j]+=returns[j+1]
        return zip(states,actions,returns)

    def reset(self):
        self.current = 0

    def isempty(self):
        return bool(self.current)
