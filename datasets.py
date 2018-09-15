import os
import torch as ch
import torch.utils.data as Data
import librosa
import numpy as np
import string


class LJSpeechDataset(Data.Dataset):
    def __init__(self,root = '../LJSpeech-1.1',ttmel=1):
        self.ttmel = ttmel
        self.csvpath = os.path.join(root,'metadata.csv')
        self.wavdir = os.path.join(root,'wavs')
        
        # alpha = string.ascii_lowercase + '?!:;,.- \"()'+'\n'+"'"
        self.alpha = string.ascii_lowercase + ',.- \"'
        self.i2c = dict(enumerate(self.alpha))
        self.c2i = dict((c,i) for i,c in enumerate(self.alpha))
        self.alpha = set(self.alpha)
        
        with open(self.csvpath) as F:
            lines = F.read().split('\n')

        split = [l.split('|') for l in lines]
        split = [s for s in split if len(s) == 3]
        split = [(duid,_,dtxt.lower()) for duid,_,dtxt in split]
        self.valid = [(duid,_,dtxt) for duid,_,dtxt in split 
                 if sum(c in self.alpha for c in dtxt) == len(dtxt)]
        self.valid = np.array(self.valid)
        
#         # Lshapes.max(axis=0),Sshapes.max(axis=0),Yshapes.max(axis=0)
#         #(array([180]), array([ 80, 217]), array([513, 868]))
#         self.Llen = 180
#         self.Slen = 217
#         self.Ylen = 868
        
    def __len__(self):
        return len(self.valid)
    
    def __getitem__(self,idx):
        duid,_,dtxt = self.valid[idx]
        wavpath = os.path.join(self.wavdir,duid+'.wav')

        nFFT = 1024
        hopL = 256
        nMel = 80
        gamma,eta = 0.6,1.3
        # Lshapes.max(axis=0),Sshapes.max(axis=0),Yshapes.max(axis=0)
        #(array([180]), array([ 80, 217]), array([513, 868]))
        Lshape = (180,)
        Sshape = (80,217)
        Yshape = (513,868)
        
        def padZero(tensor,targetLen):
            if tensor.shape[-1] >= targetLen: return tensor[...,:targetLen]
            padDim = list(tensor.shape)
            padDim[-1] = max(0,targetLen-padDim[-1])
            return ch.cat((tensor.type(ch.float),
                           ch.zeros(*padDim).type(ch.float)),
                          dim=-1)
        
        audio,rate = librosa.load(wavpath)
        
        Y = librosa.core.stft(audio,n_fft=nFFT,hop_length=hopL)
        # print('total phase:', np.sum(np.abs(np.angle(Y)))) # confirm phase in stft
        Y = Y[:,:Y.shape[1]//4 * 4] # normalize length to mult of 4
        Y = np.abs(Y) # get stft magnitude
        Y = (Y/np.max(Y))**gamma # normalize w/ preemphasis factor gamma    

        S = librosa.feature.melspectrogram(audio,n_fft=nFFT,hop_length=hopL,n_mels=nMel)
        S = S[:,3::4]  # b/c deconv non causal??
        S = (S/np.max(S))**gamma
        
        if self.ttmel: #txt2mel
            S = padZero(ch.from_numpy(S),217)
            Y = padZero(ch.from_numpy(Y),868)
        else: #ssrn trains in batches of 64 to save mem
            if S.shape[1] > 64:
                i = np.random.randint(0,S.shape[1]-64+1)
                S = ch.from_numpy(S[:,i:i+64])
                Y = ch.from_numpy(Y[:,4*i:4*i+256])
            else:
                S = padZero(ch.from_numpy(S),64)
                Y = padZero(ch.from_numpy(Y),256)
        S = S.type(ch.float)
        Y = Y.type(ch.float)

        L = np.array([self.c2i[c] for c in dtxt])
        L = padZero(ch.from_numpy(L),180)
        L = L.type(ch.long)

        return L,S,Y
