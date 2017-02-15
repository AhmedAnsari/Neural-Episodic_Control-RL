# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:05:34 2016

@author: ghulamahmedansari
"""

import gym
import numpy as np
from numba import jit
import scipy.misc.imresize as imresize

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
            z[i,j] = max(x[i,j],y[i,j])
    return z

class Environment(object):
    def __init__(self, config):
        self.START_NEW_GAME = True
        self._env = gym.make(config.GAME)
        self._env.reset()
        self.DISPLAY = config.DISPLAY
        self.config = config

    # reset environment
    def reset(self):
        self._env.reset()
        self.terminal = False

    # preprocess raw image to 84*84 Y channel(luminance)
    def preprocess(self):
        self._screen = grayscale(self._screen_)
        self._screen = imresize(self._screen,(84,84))

    def step(self,action):
        self._screen_, self.reward, self.terminal, _ = self._env.step(action)

    def action_size(self):
        return self._env.action_space.n

    def render(self):
        if self.DISPLAY ==True:
            self._env.render()

    def close_render(self):
        if self.DISPLAY ==True:
            self._env.render(close=True)

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

        out = np.zeros(prevframe.shape,dtype = np.uint8)
        if not terminal:
            out = maximage(prevframe,observation,out)

        self._screen_ = out

        self.preprocess()

        clip_reward = 0

        if self.config.clipR:
            clip_reward = np.clip(Reward,-1,1)

        if terminal:
            self.START_NEW_GAME = True

        return self._screen, action, clip_reward, terminal