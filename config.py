#Configurations for Atari Games
import os
class Config:
    def __init__(self):
        self.NUM_EPOCHS = 20000000 #max number of epochs
        self.T = 10000 #max episode length
        self.k = 11 #Knearest neighbout
        self.K = 4 #frame skip
        self.noop = 30
        self.memorysize = 1000000
        self.epsilon = 0.005
        self.final_size = 64
        self.action_set_size = None

    def getaction_set_size(self,N):
        self.action_set_size = N

