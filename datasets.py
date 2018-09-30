import os
import torch as ch
import torch.utils.data as Data
import librosa
import numpy as np
import hyperparams


class LJSpeechDataset(Data.Dataset):
    def __init__(self,root = '../LJSpeech-1.1',ttmel=1,
                 params=hyperparams.Hyperparams()):
        self.ttmel = ttmel
        self.params = params
        self.csvpath = os.path.join(root,'metadata.csv')
        self.wavdir = os.path.join(root,'wavs')
        
        with open(self.csvpath) as F:
            lines = F.read().split('\n')

        split = [l.split('|') for l in lines]
        split = [s for s in split if len(s) == 3]
        split = [(duid,_,dtxt.lower()) for duid,_,dtxt in split]
        self.valid = [(duid,_,dtxt) for duid,_,dtxt in split 
                 if sum(c in params.c2i for c in dtxt) == len(dtxt)]
        self.valid = np.array(self.valid)
        
#         # Lshapes.max(axis=0),Sshapes.max(axis=0),Yshapes.max(axis=0)
#         #(array([180]), array([ 80, 217]), array([513, 868]))
#         self.Llen = 180
#         self.Slen = 217
#         self.Ylen = 868
        
    def __len__(self):
        return len(self.valid)
    
    def __getitem__(self,idx):
        params = self.params #fix later
        duid,_,dtxt = self.valid[idx]
        wavpath = os.path.join(self.wavdir,duid+'.wav')

        # Lshapes.max(axis=0),Sshapes.max(axis=0),Yshapes.max(axis=0)
        #(array([180]), array([ 80, 217]), array([513, 868]))
        # Lshape = (180,)
        # Sshape = (80,217)
        # Yshape = (513,868)
        
        def padZero(tensor,targetLen,i=0):
            if tensor.shape[-1] >= targetLen: return tensor[...,:targetLen],0
            padDim = list(tensor.shape)
            padDim[-1] = max(0,targetLen-padDim[-1])
            i = np.random.randint(0,padDim[-1]) if i == None else i
            pad = ch.zeros(*padDim).type(ch.float)
            return ch.cat([t for t in (pad[...,:i],
                                       tensor.type(ch.float),
                                       pad[...,i:])
                           if np.prod(t.shape)],
                          dim=-1),i
        
        audio,rate = librosa.load(wavpath)
        
        Y = librosa.core.stft(audio,
                              n_fft=params.nFFT,
                              hop_length=params.hopL)
        # print('total phase:', np.sum(np.abs(np.angle(Y)))) # confirm phase in stft
        Y = Y[:,:Y.shape[1]//4 * 4] # normalize length to mult of 4
        Y = np.abs(Y) # get stft magnitude
        Y = (Y/np.max(Y))**params.gamma # normalize w/ preemphasis factor gamma    

        S = librosa.feature.melspectrogram(audio,
                                           n_fft=params.nFFT,
                                           hop_length=params.hopL,
                                           n_mels=params.nMel)
        S = S[:,3::4]  # b/c deconv non causal??
        S = (S/np.max(S))**params.gamma
        
        if self.ttmel: #txt2mel
            # shift signal to random i if pad == 2 else specify i to be 0 (no shift)
            i = None if params.pad == 2 else 0
            S,i = padZero(ch.from_numpy(S),217,i=i)
            Y,_ = padZero(ch.from_numpy(Y),4*217,i=4*i)
        else: #ssrn trains in batches of 64 to save mem
            if S.shape[1] > 64:
                i = np.random.randint(0,S.shape[1]-64+1)
                S = ch.from_numpy(S[:,i:i+64])
                Y = ch.from_numpy(Y[:,4*i:4*i+256])
            else:
                S,i = padZero(ch.from_numpy(S),64)
                Y,_ = padZero(ch.from_numpy(Y),256)
        S = S.type(ch.float)
        Y = Y.type(ch.float)

        L = np.array([params.c2i[c] for c in dtxt])
        L,_ = padZero(ch.from_numpy(L),180)
        L = L.type(ch.long)

        return L,S,Y,i
