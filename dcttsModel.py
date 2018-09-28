import torch as ch
from hyperparams import Hyperparams as params

class C(ch.nn.Module):
    def __init__(self,o,i,k,d,causal,s=1):
        super(C,self).__init__()
        self.causal = causal
        assert (k-1)%2 == 0 
        assert k > 0
        if causal:
            self.pad = (k-1)*d
        else:
#             print('filter',k,'dilation',d,'total pad',(k-1)*d,'half pad',(k-1)*d//2)
            self.pad = (k-1)*d // 2 
        self.dilation = d
        self.k = k
        self.o = o
        self.i = i
        if params.sep == 3:# and o == i:
            sqz = 4
            self.reduce = ch.nn.Conv1d(out_channels=i//sqz,in_channels=i,
                                  kernel_size=1)
            self.conv = ch.nn.Conv1d(out_channels=o//sqz, in_channels=i//sqz,
                    kernel_size=k, dilation=d, stride=s, padding=self.pad)
            self.expand = ch.nn.Conv1d(out_channels=o,in_channels=o//sqz,
                                  kernel_size=1)
            ch.nn.init.kaiming_normal_(self.reduce.weight.data)
            ch.nn.init.kaiming_normal_(self.conv.weight.data)
            ch.nn.init.kaiming_normal_(self.expand.weight.data)
        elif params.sep in (1,2) and k > 1:
            g = 4
            if params.sep == 2 and o%g == 0 and i%g == 0: chanGroups = g
            else: chanGroups = 1
            self.depthwise = ch.nn.Conv1d(out_channels=i, in_channels=i,
                        kernel_size=k, dilation=d, stride=s,
                        padding=self.pad, groups=i)
            self.pointwise = ch.nn.Conv1d(out_channels=o, in_channels=i,
                                          kernel_size=1, groups=chanGroups)
            ch.nn.init.kaiming_normal_(self.depthwise.weight.data)
            ch.nn.init.kaiming_normal_(self.pointwise.weight.data)
#             self.conv = lambda X: self.pointwise(self.depthwise(X))
        else:
            self.conv = ch.nn.Conv1d(out_channels=o, in_channels=i,
                    kernel_size=k, dilation=d, stride=s, padding=self.pad)
            ch.nn.init.kaiming_normal_(self.conv.weight.data)
        # layer norm over channel
        if params.norm == 2: self.norm = ch.nn.LayerNorm((o,))
        # batch norm over channel
        elif params.norm == 1: self.norm = ch.nn.BatchNorm1d(num_features=o)
        if params.dropout: self.dropout = ch.nn.Dropout(p=params.dropout)
    
    def forward(self,X):
        if params.dropout: X = self.dropout(X)
        if params.sep == 3:# and self.o == self.i:
            O = self.expand(self.conv(self.reduce(X)))
        elif params.sep in (1,2) and k > 1:
            O = self.pointwise(self.depthwise(X))
        else:
            O = self.conv(X)
#         O = self.conv(X)
#         print(X.shape,O.shape,self.pad)
        O = O[:,:,:-self.pad] if self.causal and self.pad else O
        if params.norm == 2: # layer norm over channel
            O = self.norm(O.permute((0,2,1))).permute((0,2,1))
        elif params.norm == 1: # batch norm over channel
            O = self.norm(O)
        return O

class D(ch.nn.Module):
    def __init__(self,o,i,k,d,causal=0,s=2):
        super(D,self).__init__()
        self.tconv = ch.nn.ConvTranspose1d(out_channels=o, in_channels=i, 
                       kernel_size=k, dilation=d, stride=s)
        ch.nn.init.kaiming_normal_(self.tconv.weight.data)
    
    def forward(self,X):
        return self.tconv(X)

class HC(ch.nn.Module):
    def __init__(self,o,i,k,d,causal,s=1):
        assert o == i
        super(HC,self).__init__()
        self.o = o
        self.conv = C(2*o,i,k,d,causal,s)

    def forward(self,X):
        H = self.conv(X)
        H1,H2 = H[:,:self.o,:],H[:,self.o:,:]
        G = ch.sigmoid(H1)
        return G*H2 + (1-G)*X

class TextEnc(ch.nn.Module):
    def __init__(self,d,e,c2i):
        super(TextEnc,self).__init__()
        c = 0 # non causal
        self.embed = ch.nn.Embedding(len(c2i),e)
        ch.nn.init.kaiming_normal_(self.embed.weight.data)
        layers = [C(2*d,e,1,1,c),ch.nn.ReLU(),C(2*d,2*d,1,1,c)]
        for _ in range(2):
            layers += [HC(2*d,2*d,3,3**ldf,c) for ldf in range(4)]
        layers += [HC(2*d,2*d,3,1,c) for _ in range(2)]
        layers += [HC(2*d,2*d,1,1,c) for _ in range(2)]
        self.seq = ch.nn.Sequential(*layers)
    
    def forward(self,L):
        # permute b/c next layer expects dims to be [batch,embed,seq]
        # output of embed layer is [batch,seq,embed]
#         print(L.shape,self.embed(L).shape)
#         print(self.embed(L).permute(0,2,1).shape)
        return self.seq(self.embed(L).permute(0,2,1))

class AudioEnc(ch.nn.Module):
    def __init__(self,d,F):
        super(AudioEnc,self).__init__()
        c = 1 # causal
        layers = [C(d,F,1,1,c),ch.nn.ReLU(),C(d,d,1,1,c),ch.nn.ReLU(),C(d,d,1,1,c)]
        for _ in range(2):
            layers += [HC(d,d,3,3**ldf,c) for ldf in range(4)]
        layers += [HC(d,d,3,3,c) for _ in range(2)]
        self.seq = ch.nn.Sequential(*layers)
        
    def forward(self,S):
        return self.seq(S)

class Text2Mel(ch.nn.Module):
    def __init__(self,d,e,c2i,F):
        super(Text2Mel,self).__init__()
        self.d = d
        self.textEnc = TextEnc(d=d,e=e,c2i=c2i)
        self.audioEnc = AudioEnc(d,F)
        self.audioDec = AudioDec(d,F)
    
    def forward(self,L,S):
        KV = self.textEnc(L)
        K,V = KV[:,:self.d,:],KV[:,self.d:,:]
        Q = self.audioEnc(S[:,:,:])
#         print('K',K.shape,'V',V.shape,'Q',Q.shape)
        A = ch.nn.Softmax(dim=1)(ch.matmul(ch.transpose(K,-1,-2),Q) / self.d**0.5)
        R = ch.matmul(V,A)
        Rp = ch.cat([R,Q],dim=1)
        S = self.audioDec(Rp)
#         print('R',R.shape,'Q',Q.shape)
        return S,A

class AudioDec(ch.nn.Module):
    def __init__(self,d,F):
        super(AudioDec,self).__init__()
        s = 1 # causal
        layers = [C(d,2*d,1,1,s)]
        for _ in range(1): #?
            layers += [HC(d,d,3,3**ldf,s) for ldf in range(4)]
        layers += [HC(d,d,3,1,s) for _ in range(2)]
        for _ in range(3): 
            layers += [C(d,d,1,1,s),ch.nn.ReLU()]
        layers += [C(F,d,1,1,s),ch.nn.Sigmoid()]
        self.seq = ch.nn.Sequential(*layers)
    
    def forward(self,Rp):
        return self.seq(Rp)

class SSRN(ch.nn.Module):
    def __init__(self,c,F,Fp):
        super(SSRN,self).__init__()
        s = 0 # non causal
        layers = [C(c,F,1,1,s)]
        for _ in range(1): #?
            layers += [HC(c,c,3,1,s),HC(c,c,3,3,s)]
        for _ in range(2):
            layers += [D(c,c,2,1),HC(c,c,3,1,s),HC(c,c,3,3,s)]
        layers += [C(2*c,c,1,1,s)]
        layers += [HC(2*c,2*c,3,1,s) for _ in range(2)]
        layers += [C(Fp,2*c,1,1,s)]
        for _ in range(2):
            layers += [C(Fp,Fp,1,1,s),ch.nn.ReLU()]
        layers += [C(Fp,Fp,1,1,s),ch.nn.Sigmoid()]
        self.seq = ch.nn.Sequential(*layers)
    
    def forward(self,Y):
        return self.seq(Y)