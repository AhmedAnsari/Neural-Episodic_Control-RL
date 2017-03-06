#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:02:28 2017

@author: ghulamahmedansari
"""
from config import Config
from environment import Environment,Eval_Environment
from tqdm import tqdm
from controller import Control
from buffer import Buffer
from plotter import Plotter
import numpy as np

def evaluate(env,config,brain,curr_step,plt):
    episode_length = 0
    env.frame_history = 0
    cumulative_r = 0
    pbar = tqdm(total = config.NUM_EVAL_STEPS, desc='Testing Progress')
    ep_r = []
    while env.frame_history <= config.NUM_EVAL_STEPS:
        past_num_frames = env.frame_history
        if episode_length == 0:
            env.reset()
            s,a,r,t = env.act(0)
            episode_length += 1
            cumulative_r+=r
        s,a,r,t = env.act(brain.getaction(s))
        episode_length += 1
        cumulative_r+=r

        if env.START_NEW_GAME:#then epsiode ends
            episode_length = 0
            ep_r.append(cumulative_r)
            cumulative_r=0
        pbar.update(env.frame_history-past_num_frames)

        ep_avg_r = np.mean(ep_r)
        plt.writesummary(ep_avg_r)

def main():
    config = Config()
    env = Environment(config) #for training
    eval_env = Eval_Environment(config)#for testing
    num_actions = env.action_size()
    config.setaction_set_size(num_actions)
    brain = Control(config)
    plt = Plotter()
    plt.writesummary(0)
    #adding progress bar for training
    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')


    episode_buffer = Buffer(config)
    episode_length = 0

    eval_count = 1
    while(env.frame_history <= config.MAX_FRAMES):
        if env.frame_history/(config.EVAL_FREQ*eval_count) == 1:
            evaluate(eval_env,config,brain,env.frame_history,plt)#testing happens now
            eval_count+=1
        past_num_frames = env.frame_history
        #algorithm beigns now

        if episode_length == 0:
            env.reset()
            s,a,r,t = env.act(0)
            episode_buffer.add(s,a,r)
            episode_length += 1

        s,a,r,t = env.act(brain.getaction(s))
        episode_length += 1
        episode_buffer.add(s,a,r)

        if (env.START_NEW_GAME or episode_length >= config.T) and not(episode_buffer.isempty()):#then epsiode ends
            episode_values = episode_buffer.get_returns()
            brain.update_table(episode_values)
            episode_buffer.reset()
            episode_length = 0

        pbar.update(env.frame_history-past_num_frames)

    env.close_render()

if __name__ == '__main__':
    main()