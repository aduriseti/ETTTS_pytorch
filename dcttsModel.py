import torch as ch
import hyperparams
import utils

class C(ch.nn.Module):
    def __init__(self,o,i,k,d,causal,params=hyperparams.Hyperparams(),stride=1):
        super(C,self).__init__()
        self.causal = causal
        self.params = params
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
                    kernel_size=k, dilation=d, stride=stride, padding=self.pad)
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
                        kernel_size=k, dilation=d, stride=stride,
                        padding=self.pad, groups=i)
            self.pointwise = ch.nn.Conv1d(out_channels=o, in_channels=i,
                                          kernel_size=1, groups=chanGroups)
            ch.nn.init.kaiming_normal_(self.depthwise.weight.data)
            ch.nn.init.kaiming_normal_(self.pointwise.weight.data)
#             self.conv = lambda X: self.pointwise(self.depthwise(X))
        else:
            self.conv = ch.nn.Conv1d(out_channels=o, in_channels=i,
                    kernel_size=k, dilation=d, stride=stride, padding=self.pad)
            ch.nn.init.kaiming_normal_(self.conv.weight.data)
        # layer norm over channel
        if params.norm == 2: self.norm = ch.nn.LayerNorm((o,))
        # batch norm over channel
        elif params.norm == 1: self.norm = ch.nn.BatchNorm1d(num_features=o)
        if params.dropout: self.dropout = ch.nn.Dropout(p=params.dropout)
    
    def forward(self,X):
        if self.params.dropout: X = self.dropout(X)
        if self.params.sep == 3:
            O = self.expand(self.conv(self.reduce(X)))
        elif self.params.sep in (1,2) and k > 1:
            O = self.pointwise(self.depthwise(X))
        else:
            O = self.conv(X)
#         O = self.conv(X)
        O = O[:,:,:-self.pad] if self.causal and self.pad else O
        if self.params.norm == 2: # layer norm over channel
            O = self.norm(O.permute((0,2,1))).permute((0,2,1))
        elif self.params.norm == 1: # batch norm over channel
            O = self.norm(O)
        return O

class D(ch.nn.Module):
    def __init__(self,o,i,k,d,params=hyperparams.Hyperparams(),causal=0,s=2):
        super(D,self).__init__()
        self.tconv = ch.nn.ConvTranspose1d(out_channels=o, in_channels=i, 
                       kernel_size=k, dilation=d, stride=s)
        ch.nn.init.kaiming_normal_(self.tconv.weight.data)
    
    def forward(self,X):
        return self.tconv(X)

class HC(ch.nn.Module):
    def __init__(self,o,i,k,d,causal,params=hyperparams.Hyperparams(),stride=1):
        assert o == i
        super(HC,self).__init__()
        self.o = o
        self.conv = C(2*o,i,k,d,causal,params,stride)

    def forward(self,X):
        H = self.conv(X)
        H1,H2 = H[:,:self.o,:],H[:,self.o:,:]
        G = ch.sigmoid(H1)
        return G*H2 + (1-G)*X

class TextEnc(ch.nn.Module):
    def __init__(self,params=hyperparams.Hyperparams()):
        super(TextEnc,self).__init__()
        c = 0 # non causal
        d,e,alphabet = params.d,params.e,params.alphabet
        self.embed = ch.nn.Embedding(len(alphabet),e)
        ch.nn.init.kaiming_normal_(self.embed.weight.data)
        layers = [C(2*d,e,1,1,c,params),ch.nn.ReLU(),C(2*d,2*d,1,1,c,params)]
        for _ in range(2):
            layers += [HC(2*d,2*d,3,3**ldf,c,params) for ldf in range(4)]
        layers += [HC(2*d,2*d,3,1,c,params) for _ in range(2)]
        layers += [HC(2*d,2*d,1,1,c,params) for _ in range(2)]
        self.seq = ch.nn.Sequential(*layers)
    
    def forward(self,L):
        # permute b/c next layer expects dims to be [batch,embed,seq]
        # output of embed layer is [batch,seq,embed]
        return self.seq(self.embed(L).permute(0,2,1))

class AudioEnc(ch.nn.Module):
    def __init__(self,params=hyperparams.Hyperparams()):
        super(AudioEnc,self).__init__()
        c = 1 # causal
        d,F = params.d,params.F
        layers = [C(d,F,1,1,c,params),ch.nn.ReLU(),
                  C(d,d,1,1,c,params),ch.nn.ReLU(),
                  C(d,d,1,1,c,params)]
        for _ in range(2):
            layers += [HC(d,d,3,3**ldf,c,params) for ldf in range(4)]
        layers += [HC(d,d,3,3,c,params) for _ in range(2)]
        self.seq = ch.nn.Sequential(*layers)
        
    def forward(self,S):
        return self.seq(S)

class AudioDec(ch.nn.Module):
    def __init__(self,params=hyperparams.Hyperparams()):
        super(AudioDec,self).__init__()
        s = 1 # causal
        d,F = params.d,params.F
        layers = [C(d,2*d,1,1,s,params)]
        for _ in range(1): #?
            layers += [HC(d,d,3,3**ldf,s,params) for ldf in range(4)]
        layers += [HC(d,d,3,1,s,params) for _ in range(2)]
        for _ in range(3): 
            layers += [C(d,d,1,1,s,params),ch.nn.ReLU()]
        layers += [C(F,d,1,1,s,params),ch.nn.Sigmoid()]
        self.seq = ch.nn.Sequential(*layers)
    
    def forward(self,Rp):
        return self.seq(Rp)

    
class Text2Mel(ch.nn.Module):
    def __init__(self,params=hyperparams.Hyperparams(),*args,**kwargs):
        super(Text2Mel,self).__init__(*args,**kwargs)
        self.params = params
        self.textEnc = TextEnc(params)
        self.audioEnc = AudioEnc(params)
        self.audioDec = AudioDec(params)
    
    def forward(self,L,S):
        KV = self.textEnc(L)
        d = self.params.d
        K,V = KV[:,:d,:],KV[:,d:,:]
        Q = self.audioEnc(S[:,:,:])
        A = ch.nn.Softmax(dim=1)(ch.matmul(ch.transpose(K,-1,-2),Q) / self.d**0.5)
        R = ch.matmul(V,A)
        Rp = ch.cat([R,Q],dim=1)
        S = self.audioDec(Rp)
        return S,A

class SSRN(ch.nn.Module):
    def __init__(self,params=hyperparams.Hyperparams(),*args,**kwargs):
        super(SSRN,self).__init__(*args,**kwargs)
        s = 0 # non causal
        c,F,Fp = params.c,params.F,params.Fp
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