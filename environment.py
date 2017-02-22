# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:05:34 2016

@author: ghulamahmedansari
"""

import gym
import numpy as np
from numba import jit
from skimage.transform import resize
import random
from plotter import Plotter
from collections import deque

@jit
def grayscale(raw_screen):
    y =  0.2126 * raw_screen[:, :, 0] + 0.7152 * raw_screen[:, :, 1] + 0.0722 * raw_screen[:, :, 2]
    y = y.astype(np.uint8)
    return y

@jit
def maximage(x,y,z):
    M = x.shape[0]
    N = x.shape[1]
    for i in range(M):
        for j in range(N):
            for k in range(3):
                z[i,j,k] = max(x[i,j,k],y[i,j,k])
    return z

class Environment(object):
    def __init__(self, config):
        self.START_NEW_GAME = False
        self.config = config
        self._env = gym.make(config.GAME)
        self.DISPLAY = config.DISPLAY
        self.frame_history = 0
        self.A = np.random.random((config.final_size, 84*84))#Transformation matrix
        self.reset()
        self.rewards_list = deque(maxlen=2)
        self.rewards_list.append([(0,0)])
        self.plt = Plotter()
    # reset environment
    def reset(self):
        self._env.reset()
        self.START_NEW_GAME = False
        action0 = 0
        #ensure random start
        for _ in range(random.randrange(1,self.config.noopmax+1)):
            self.step(action0)
            self.frame_history -= 1 #dont count frames when performing noop for stachisticity

    # preprocess raw image to 84*84 Y channel(luminance)
    def preprocess(self):
        self._screen = grayscale(self._screen_)
        self._screen = resize(self._screen,(84,84))
        self._screen = np.dot(self.A , self._screen.reshape(84*84,1)).reshape(self.config.final_size)

    def step(self,action):
        self._screen_, self.reward, self.terminal, _ = self._env.step(action)
        self.frame_history += 1

    def action_size(self):
        return self._env.action_space.n

    def render(self):
        if self.DISPLAY ==True:
            self._env.render()

    def close_render(self):
        if self.DISPLAY ==True:
            self._env.render(close=True)

    def evaluate(self):
        s,r = self.rewards_list[-1]

        self.plt.writesummary([(s,r/float(s))])

    def act(self,action):
        Reward = 0
        start_lives = self._env.ale.lives()
        terminal = False

        for _ in xrange(self.config.K):
            #to store difference of frames
            if _ == self.config.K - 2:
                prevframe = self._screen_.copy()

            self.render()
            self.step(action)
            observation,localreward,terminal = self._screen_,self.reward,self.terminal
            Reward += localreward

            if start_lives > self._env.ale.lives():
                Reward -= 1.0
                terminal = True

            if terminal:
                break

        out = np.zeros(observation.shape,dtype = np.uint8)
        if not terminal:
            maximage(prevframe,observation,out)

        self._screen_ = out

        self.preprocess()

        clip_reward = 0

        if self.config.clipR:
            clip_reward = np.clip(Reward,-1,1)

        if terminal:
            self.START_NEW_GAME = True

        #for evaluations of avg_r
        self.rewards_list.append((self.frame_history,self.rewards_list[-1][1]+clip_reward))
        self.evaluate()

        return self._screen, action, clip_reward, terminal
