#Configurations for Atari Games

class Config:
    def __init__(self):
        self.MAX_FRAMES = 20000000 #max number of epochs
        self.T = 10000 #max episode length
        self.k = 11 #Knearest neighbout
        self.K = 4 #frame skip
        self.noop = 30
        self.memorysize = 1000000
        self.epsilon = 0.005
        self.final_size = 64
        self.action_set_size = None
        self.DISPLAY = False
        self.GAME = 'Breakout-v0'

    def setaction_set_size(self,N):
        self.action_set_size = N

