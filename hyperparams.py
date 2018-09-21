import string

class Hyperparams:
    
    # alphabet = string.ascii_lowercase + '?!:;,.- \"()'+'\n'+"'"
    alphabet = string.ascii_lowercase + ',.- \"'
    i2c = dict(enumerate(alphabet))
    c2i = dict((c,i) for i,c in enumerate(alphabet))

    # 0: no sep, 1: depthwise sep, 2: super sep
    sep = 0
    # model width multiple - determines # of channels at each layer
    alpha = 1
    # controls dropout after conv layers
    dropout = 0
    # controls normalization
    # 0: no norm, 1: batch norm, 2: channel norm, 3: weight norm, 4: instance norm, 5: group norm
    norm = 0
    # learning rate
#     lr = 1e-3 # from that korean guys hyperparameters: https://github.com/Kyubyong/dc_tts
    lr = 2e-4 # from the original paper: https://arxiv.org/abs/1710.08969
    chunk = 1 # generate 1 timestep per autoregression
    
    # LR TOO HIGH
    # HYPERPARAMS {'sep': 0, 'alpha': 1, 'dropout': 0, 'norm': 0, 'lr': 0.001, 'chunk': 1}
    # HYPERPARAMS {'sep': 0, 'alpha': 1, 'dropout': 1, 'norm': 0, 'lr': 0.001, 'chunk': 1}
    
    
    d = int(256*alpha)
    e = int(128*alpha)
    c = int(512*alpha)
    F = 80
    Fp = 513
    
    g=0.2
    
    # Adam params - not including lr - which we vary
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