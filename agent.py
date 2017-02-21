#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:02:28 2017

@author: ghulamahmedansari
"""
from config import Config
from environment import Environment
from tqdm import tqdm
from controller import Control
from buffer import Buffer

def main():
    config = Config()
    env = Environment(config)
    num_actions = env.action_size()
    config.setaction_set_size(num_actions)

    brain = Control(config)
    #adding progress bar for training
    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')


    episode_buffer = Buffer(config)
    episode_length = 0
    while(env.frame_history <= config.MAX_FRAMES):

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

        pbar.update(config.K)


    env.close_render()

if __name__ == '__main__':
    main()