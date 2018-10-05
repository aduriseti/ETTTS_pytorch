import string

class Hyperparams:
        # LR TOO HIGH
    # HYPERPARAMS {'sep': 0, 'alpha': 1, 'dropout': 0, 'norm': 0, 'lr': 0.001, 'chunk': 1}
    # HYPERPARAMS {'sep': 0, 'alpha': 1, 'dropout': 1, 'norm': 0, 'lr': 0.001, 'chunk': 1}
    
    # UNSTABLE
#     HYPERPARAMS sep:0|alpha:1|dropout:0|norm:0|lr:0.0002|chunk:1
# LOADED EPOCH 10, LOSS 0.031800729800849706, BEST LOSS 0.031800729800849706 FROM


    # BEST YET - INTELLIGIBLE SPEECH
    # dropout definately helps generalize to A-team intro
    # HYPERPARAMS sep:0|alpha:1|dropout:0.05|norm:2|lr:0.001|chunk:1
# LOADED EPOCH 102, LOSS 0.024689697037770676, BEST LOSS 0.024689697037770676 FROM
# HYPERPARAMS sep:0|alpha:1|dropout:0.05|norm:2|lr:0.001|chunk:1
# LOADED EPOCH 31, LOSS 0.10023909083275653, BEST LOSS 0.10023909083275653 FROM
# 100/100 [00:02<00:00, 34.42it/s] GPU
# 100/100 [00:17<00:00, 5.66it/s] CPU

#     HYPERPARAMS sep:0|alpha:1|dropout:0|norm:2|lr:0.001|chunk:1
#     LOADED EPOCH 99, LOSS 0.0231524246063695, BEST LOSS 0.0231524246063695 FROM
#     LOADED EPOCH 29, LOSS 0.09947643938349254, BEST LOSS 0.09947643938349254 FROM



# HYPERPARAMS sep:1|alpha:1|dropout:0.05|norm:2|lr:0.001|chunk:1
# 100/100 [00:02<00:00, 44.49it/s] GPU
# 100/100 [00:33<00:00, 2.96it/s] CPU

# HYPERPARAMS sep:0|alpha:1|dropout:0.05|norm:2|lr:0.001|chunk:4
# LOADED EPOCH 99, LOSS 0.025080004773700416, BEST LOSS 0.025080004773700416 FROM
# 25/25 [00:02<00:00, 8.87it/s] CPU
# 25/25 [00:00<00:00, 50.81it/s] GPU
    tuneable = ['sep','alpha','dropout','norm','lr','chunk','sqz','pad',
                'reversedDilation']
    def __init__(self,sep=0,alpha=1,dropout=0.05,
                 norm=2,lr=1e-3,chunk=1,sqz=4,pad=0,
                 reversedDilation=0):
        # 0: no sep, 1: depthwise sep, 2: super sep, 3: bottleneck
        self.sep = sep
        # model width multiple - determines # of channels at each layer
        self.alpha = alpha
        # controls dropout before conv layers
        # # https://github.com/Kyubyong/dc_tts: 0.05
        self.dropout = dropout
        # controls normalization
        # 0: no norm, 1: batch norm, 2: channel norm, 3: weight norm, 4: instance norm, 5: group norm
        self.norm = norm
        # learning rate
#         lr = 1e-3 # from that korean guys hyperparameters: https://github.com/Kyubyong/dc_tts
#         lr = 2e-4 # from the original paper: https://arxiv.org/abs/1710.08969
        self.lr = lr
#         number of spectrogram banks to generate per iteration - 1 in original paper
        self.chunk = chunk
        # channel divider for bottleneck convolution layer - only used if bottleneck conv used
        self.sqz = 4 if sep == 3 else None
        # determines how spectrogram data is zero-padded
        # 0: pad from right, 1: pad from left, 2: pad a random amount from both directions
        self.pad = pad
        
        self.reversedDilation = reversedDilation
        
        self.paramDict = dict((p,self.__dict__[p]) for p in self.tuneable if self.__dict__[p] != None)
        
        self.d = int(256*alpha)
        self.e = int(128*alpha)
        self.c = int(512*alpha)
        self.F = 80
        self.Fp = 513
    
    # alphabet = string.ascii_lowercase + '?!:;,.- \"()'+'\n'+"'"
    # 'N' indicates null character and is mapped to 0 for zero padding
    alphabet = 'N' + string.ascii_lowercase + ',.- \"'
    i2c = dict((i,c) for i,c in enumerate(alphabet))
    c2i = dict((c,i) for i,c in enumerate(alphabet))
    
    g=0.2
    
    # Adam params - not including lr - which I vary
    b1 = 0.5
    b2 = 0.9
    eps = 1e-6
    
    logevery = 200
    
    dropout_rate = 0.1
    masking = False
    
    gamma = 0.6
    eta = 1.3
    nFFT = 1024
    hopL = 256
    nMel = 80
    rate = 22050
    
    Lshape = (180,)
    Sshape = (80,217)
    Yshape = (513,868)