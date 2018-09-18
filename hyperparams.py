import string

class Hyperparams:
    
    # alphabet = string.ascii_lowercase + '?!:;,.- \"()'+'\n'+"'"
    alphabet = string.ascii_lowercase + ',.- \"'
    i2c = dict(enumerate(alphabet))
    c2i = dict((c,i) for i,c in enumerate(alphabet))

    
    sep = 0
    alpha = 0.5
    d = int(256*alpha)
    e = int(128*alpha)
    c = int(512*alpha)
    F = 80
    Fp = 513
    
    g=0.2
    
    lr = 2e-4
    init_lr=2e-4
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