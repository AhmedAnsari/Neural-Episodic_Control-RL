#Configurations for Atari Games

class Config:
    def __init__(self):
        self.MAX_FRAMES = 20000000#20000000 #max number of epochs
        self.NUM_EVAL_STEPS  = 10000
        self.EVAL_FREQ = 50000

        self.T = 1000 #max episode length
        self.k = 11 #Knearest neighbout
        self.K = 4 #frame skip
        self.noopmax = 30
        self.memory_size = 1000000
        self.epsilon = 0.005
        self.final_size = 64
        self.action_set_size = None
        self.DISPLAY = False
        self.GAME = 'MsPacman-v0'
        self.clipR = True
        self.KNNmethod = "ANNOY"
        self.n_trees = 7
        self.maxindexsize = self.memory_size + 10000
        self.Use_Mylru = True
        if self.Use_Mylru == True and self.KNNmethod == "FLANN":
            self.KNNmethod = "ANNOY"

    def setaction_set_size(self,N):
        self.action_set_size = N

