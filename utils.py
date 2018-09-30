import torch as ch
import os
import hyperparams
import matplotlib.pyplot as plt

# # 0: no sep, 1: depthwise sep, 2: super sep
# sep = 0
# # model width multiple - determines # of channels at each layer
# alpha = 0.5
# # controls dropout after conv layers
# dropout = False
# # controls normalization
# # 0: no norm, 1: batch norm, 2: channel norm, 3: weight norm, 4: instance norm, 5: group norm
# norm = 1
# # learning rate
# lr = 2e-4 # from that korean guys hyperparameters
# paramDict = {'sep':params.sep,'alpha':params.alpha,
#              'dropout':params.dropout,'norm':params.norm,
#              'lr':params.lr,'chunk':params.chunk}

def saveChkpt(state,model,root='.',params=hyperparams.Hyperparams()):
    paramDict = dict((p,params.__dict__[p]) for p in params.tuneable if params.__dict__[p] != None)
    chkptDirNm = "|".join("{}:{}".format(k,v) for k,v in paramDict.items())
    chkptDir = os.path.join(root,chkptDirNm)
    chkptNm = model+'Chkpt.pth.tar'
    bestNm = model+'Best.pth.tar'
    if not os.path.exists(chkptDir): os.makedirs(chkptDir)
    savePaths = [os.path.join(chkptDir,chkptNm),os.path.join(root,chkptNm)]
    # found copying files using shutil unreliable
    if state['lossHist'][-1] <= state['bestLoss']: 
        savePaths += [os.path.join(chkptDir,bestNm),os.path.join(root,bestNm)]
    [ch.save(state,path) for path in savePaths]
    print("HYPERPARAMS",chkptDirNm)
    print("SAVED EPOCH {}, LOSS {}, BEST LOSS {} TO {}".format(state['epoch'],state['lossHist'][-1],state['bestLoss'],savePaths))


def loadChkpt(network,optimizer,model,dev='cpu',root='.', 
              params=hyperparams.Hyperparams()):
    chkptDirNm = "|".join("{}:{}".format(k,v) for k,v in params.paramDict.items())
    chkptDir = os.path.join(root,chkptDirNm)
    chkptNm = model+'Chkpt.pth.tar'
    bestNm = model+'Best.pth.tar'
    loadPaths = [os.path.join(chkptDir,bestNm),
                 os.path.join(chkptDir,chkptNm),
                 os.path.join(root,bestNm),
                 os.path.join(root,chkptNm)]
    print('HYPERPARAMS',chkptDirNm)
    for path in loadPaths: 
        if not os.path.exists(path): 
            print('PATH DOES NOT EXIST:',path)
            continue
        state = ch.load(path,map_location=dev)
        network.load_state_dict(state['modelState'])
        optimizer.load_state_dict(state['optimizerState'])
#         if len(state['lossHist']) > 10: plt.plot(state['lossHist'])
        print("LOADED EPOCH {}, LOSS {}, BEST LOSS {} FROM".format(state['epoch'],state['lossHist'][-1],state['bestLoss'],path))
        return state['epoch'],state['lossHist'],state['bestLoss']
    return 0,[],float('inf')


class ChkptModule(ch.nn.Module):
    def __init__(self,params=hyperparams.Hyperparams()):
        super(ChkptModule,self).__init__()

import tqdm
class ModelWrapper:
    def __init__(self,network,optimizer,lossFun,loader,modelName,
                 dev='cpu',root='.',params=hyperparams.Hyperparams()):
        self.network = network
        self.optimizer = optimizer
        self.lossFun = lossFun
        self.loader = loader
        self.modelName = modelName
        self.dev = dev
        self.root = root
        self.params = params
        self.startEpoch = 0
        self.lossHist = []
        self.bestLoss = float('inf')
        print('INITIALIZED {} WITH PARAMS'.format(modelName),text2MelParams.paramDict)
        
    def evaluate(self,evalFun):
        pass
#         evalFun
    
    def train(self,numEpochs=50,progressBar=tqdm.tqdmNotebook):
        for epoch in range(self.startEpoch,self.startEpoch+numEpochs):
            print("EPOCH",epoch)
            epochLoss = []
            for step,batch in progressBar(enumerate(self.loader)):
                batchL,batchS,batchY,batchI = batch
                bL = ch.autograd.Variable(batchL.to(self.dev))
                bS = ch.autograd.Variable(batchS.to(self.dev))
                bY = ch.autograd.Variable(batchY.to(self.dev))
                bI = ch.autograd.Variable(batchI.to(self.dev))
                loss = self.lossFun(self.network,bL,bS,bY,bI)
                epochLoss.append(loss.data.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.lossHist.append(np.mean(epochLoss))
            print('epoch',epoch,'total',lossHist[-1])
            self.bestLoss = min(lossHist[-1],self.bestLoss)
            self.save()
            self.evaluate()
            
    def save(self):
        state = {
            'epoch': len(self.lossHist),
            'modelState': self.network.state_dict(),
            'lossHist': self.lossHist,
            'bestLoss': self.bestLoss,
            'optimizerState': self.optimizer.state_dict() 
        }
        saveChkpt(state,model=self.modelName)
    
    def load(self):
        self.epoch,self.lossHist,self.bestLoss = loadChkpt(self.network,self.optimizer,self.modelName,self.dev,self.root,self.params)
        
    def to(self,dev):
        self.dev = dev
        return self
    