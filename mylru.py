#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:28:28 2017

@author: ghulamahmedansari
"""
import numpy as np
from annoy import AnnoyIndex
import sys
#from kgraph import Kgraph

class MylruMem(object):
    def __init__(self,config):
        self.config = config
        self.dimension = self.config.final_size
        self.memory_size = config.memory_size
        if self.config.KNNmethod == "ANNOY":
            self.statetree = [AnnoyIndex(self.dimension) for action in range(self.config.action_set_size)]
        elif self.config.KNNmethod == "KGRAPH":
            self.statetree = [Kgraph(config) for action in range(self.config.action_set_size)]
        self.count = [0]*self.config.action_set_size
        self.age_history = [{} for action in range(self.config.action_set_size)]
        self.Qtable = [{} for action in range(self.config.action_set_size)]
        self.maxindexsize = self.config.maxindexsize
        self.deleted_states = [[] for action in range(self.config.action_set_size)]
        self.timestep = 0
        self.Flag_Build_New_Tree = [True for action in range(self.config.action_set_size)]
        #for optimization
        self.current = None

    def update_age(self,state,action):
        self.age_history[action][state] = self.timestep
        self.timestep+=1

    def getoldest(self,action):
        states = self.age_history[action].keys()
        ages = map(lambda x: self.age_history[action][x],states)
        return states[np.argmin(ages)]

    def _add(self,state,action,Return):#primitive function to add a new element
        if self.count[action]==0:
            self.statetree[action].add_item(0,state)
            self.Qtable[action][state] = Return
            self.count[action]+=1

        elif self.count[action]>0 and self.count[action]<self.memory_size:
            self.statetree[action].add_item(self.count[action],state)
            self.Qtable[action][state] = Return
            self.count[action]+=1

        elif self.count[action]>=self.memory_size and self.count[action]<self.maxindexsize:
            self.statetree[action].add_item(self.count[action],state)
            self.Qtable[action][state] = Return

            todeletestate = self.getoldest(action)
            self.deleted_states[action].append(todeletestate)
            self.age_history[action][todeletestate] = sys.maxint #to ensure dont delete same state again

            self.count[action]+=1

        elif self.count[action] == self.maxindexsize:
            for s in self.deleted_states[action]:
                del self.Qtable[action][s]
                del self.age_history[action][s]
                self.count[action]-=1
            #now remake the tree
            if self.config.KNNmethod == "ANNOY":
                self.statetree[action] = AnnoyIndex(self.dimension)
                _count = 0
                for s in self.Qtable[action].keys():
                    self.statetree[action].add_item(_count,s)
                    _count+=1
                assert _count == self.count[action]
            elif self.config.KNNmethod == "KGRAPH":
                self.statetree[action] = Kgraph(self.config)
                _count = 0
                for s in self.Qtable[action].keys():
                    self.statetree[action].add_item(_count,s)
                    _count+=1
                assert _count == self.count[action]
            self.deleted_states[action] = []
            self._add(state,action,Return)#now call again after clearing buffer
        else:
            print "ERROR: Out of Memory Bounds"
            assert 0 == 1

        self.update_age(state,action)

    def add(self,state,action,Return):#superior function to add an element
        if type(state)!=tuple:
            state = tuple(state.astype(np.int32))

        if not self.isin(state,action,self.Qtable[action]):
#        if state not in self.Qtable[action].keys():
            self._add(state,action,Return)
            self.Flag_Build_New_Tree[action] = True

        elif self.isin(state,action,self.Qtable[action]) and self.isin(state,action,self.deleted_states[action]):
#        elif state in self.Qtable[action].keys() and state in self.deleted_states[action]:

            state = self.current
            self.Qtable[action][state] = max(self.Qtable[action][state],Return)
            self.deleted_states[action].remove(state)
            self.update_age(state,action)

            todeletestate = self.getoldest(action)
            self.deleted_states[action].append(todeletestate)
            self.age_history[action][todeletestate] = sys.maxint/2 #to ensure dont delete same state again
        else:
#        elif self.isin(state,action,self.Qtable[action]) and not self.isin(state,action,self.deleted_states[action]):
#        elif state in self.Qtable[action].keys() and state not in self.deleted_states[action]:
            state = self.current
            self.Qtable[action][state] = max(self.Qtable[action][state],Return)
            self.update_age(state,action)

#        else:
#            print "ERROR: state is both deleted and undeleted"
#            assert 0 == 1

    def buildtree(self,action):
        if self.Flag_Build_New_Tree[action] == True:
            self.statetree[action].build(self.config.n_trees)
            self.Flag_Build_New_Tree[action] = False
        return self.statetree[action]

    def getQval(self,state,action):
        state = tuple(state.astype(np.int32))
        if state in self.Qtable[action].keys():
            self.update_age(state,action)
            return self.Qtable[action][state]
        else:
            if self.count[action] > self.config.k: #to handle borderline cases
                t = self.buildtree(action)
                nearest_k_indices = t.get_nns_by_vector(state,self.config.k)
                if self.config.KNNmethod == "ANNOY":
                    states = [tuple(t.get_item_vector(i)) for i in nearest_k_indices]
                elif self.config.KNNmethod == "KGRAPH":
                    states = [tuple(np.asarray(self.statetree[action].dataset[i], dtype=np.float32)) for i in nearest_k_indices]

                result = np.mean([self.Qtable[action][s] for s in states])
                [self.update_age(s,action) for s in states]

                return result
            else:
                return 0
    def isin(self,state,action,dic):
        self.current = state
        if state in dic:
            return True
        elif self.count[action] > self.config.k:
            t = self.buildtree(action)
            nearest_k_indices = t.get_nns_by_vector(state,1,include_distances=True)
            i,d = nearest_k_indices
            i = i[0]
            d = d[0]
            if 1.0*d/np.linalg.norm(state)<=1e-9    :
                self.current = tuple(np.array(t.get_item_vector(i)).astype(np.int32))
                return True
            else:
                return False
        else:
            return False
