#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:36:32 2017

@author: ghulamahmedansari
"""

import pykgraph


class MylruMem(object):
    def __init__(self,config):
        self.dataset = []
    def add_item(self,i,item):
        self.dataset.append(item)
    def build(self,ntrees):
        self.index = pykgraph.KGraph(self.dataset, 'euclidean')
        self.index.build(reverse=-1)
        self.ntrees = ntrees
    def get_nns_by_vector(self,vector,k):
        return self.index.search(vector, k, P=100)[0]