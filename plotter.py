#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:23:44 2017

@author: ghulamahmedansari
"""

import tensorflow as tf


class Plotter(object):
    def __init__(self):
        self.reward = tf.placeholder('float32',name='Average_Reward')

        self.sess = tf.Session()

        self.writer = tf.train.SummaryWriter('./logs', graph=tf.get_default_graph())

        tf.initialize_all_variables().run(session = self.sess)

        tf.scalar_summary("Avg R", self.reward)

        self.summary_op = tf.merge_all_summaries()

    def writesummary(self,rewardlist):
        for s,r in rewardlist:
            summary = self.sess.run([self.summary_op], feed_dict={self.reward : r})

            # write log
            self.writer.add_summary(summary[0], s)
