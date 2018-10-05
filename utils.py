import torch as ch
import numpy as np
import tqdm
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
    chkptDirNm = "|".join("{}:{}".format(k,v) for k,v in params.paramDict.items())
    chkptDir = os.path.join(root,chkptDirNm)
    chkptNm = model+'Chkpt.pth.tar'
    bestNm = model+'Best.pth.tar'
    if not os.path.exists(chkptDir): os.makedirs(chkptDir)
    savePaths = [os.path.join(chkptDir,chkptNm),os.path.join(root,chkptNm)]
    # found copying files using shutil unreliable
    if state['lossHist'] and state['lossHist'][-1] <= state['bestLoss']: 
        savePaths += [os.path.join(chkptDir,bestNm),os.path.join(root,bestNm)]
    [ch.save(state,path) for path in savePaths]
    print("HYPERPARAMS",chkptDirNm)
    print("SAVED EPOCH {}, LOSS {}, BEST LOSS {} TO {}".format(state['epoch'],state['lossHist'][-1] if state['lossHist'] else float('inf'),state['bestLoss'],savePaths))


def loadChkpt(network,optimizer,model,dev='cpu',root='.', 
              params=hyperparams.Hyperparams(),best=0):
    chkptDirNm = "|".join("{}:{}".format(k,v) for k,v in params.paramDict.items())
    chkptDir = os.path.join(root,chkptDirNm)
    chkptNm = model+'Chkpt.pth.tar'
    bestNm = model+'Best.pth.tar'
    loadPaths = [os.path.join(chkptDir,chkptNm),
                 os.path.join(root,chkptNm)]
    if best: loadPaths += [os.path.join(chkptDir,bestNm),
                           os.path.join(root,bestNm)]
    print('HYPERPARAMS',chkptDirNm)
    for path in loadPaths: 
        if not os.path.exists(path): 
            print('PATH DOES NOT EXIST:',path)
            continue
        state = ch.load(path,map_location=dev)
        network.load_state_dict(state['modelState'])
        optimizer.load_state_dict(state['optimizerState'])
#         if len(state['lossHist']) > 10: plt.plot(state['lossHist'])
        print("LOADED EPOCH {}, LOSS {}, BEST LOSS {} FROM {}".format(state['epoch'],state['lossHist'][-1] if state['lossHist'] else float('inf'),state['bestLoss'],path))
        return state['epoch'],state['lossHist'],state['bestLoss']
    return 0,[],float('inf')


class ChkptModule(ch.nn.Module):
    def __init__(self,params=hyperparams.Hyperparams()):
        super(ChkptModule,self).__init__()

def evalFunTemplate(network,batch): pass
def dispFunTemplate(network,batch): pass
class ModelWrapper:
    def __init__(self,network,optimizer,lossFun,loader,modelName,
                 dev='cpu',root='.',evalFun=evalFunTemplate,dispFun=dispFunTemplate):
        self.network = network
        self.optimizer = optimizer
        self.lossFun = lossFun
        self.loader = loader
        self.modelName = modelName
        self.dev = dev
        self.root = root
        self.evalFun = evalFun
        self.dispFun = dispFun
        self.params = self.network.params
        self.startEpoch = 0
        self.lossHist = []
        self.bestLoss = float('inf')
        print('INITIALIZED {} WITH HYPERPARAMS'.format(modelName),self.params.paramDict)
        print('TOTAL PARAM COUNT',sum(np.prod(p.size()) for p in network.parameters()))
        
    def evaluate(self,k=1):
        for step,batch in enumerate(self.loader):
            if step >= k: break
            batchV = [ch.autograd.Variable(t.to(self.dev)) for t in batch]
            self.evalFun(self.network,batchV)
        
    def predict(self):
        pass
    
    def disp(self,k=1):
        for step,batch in enumerate(self.loader):
            if step >= k: break
            batchV = [ch.autograd.Variable(t.to(self.dev)) for t in batch]
            self.dispFun(self.network,batchV)
    
    def train(self,numEpochs=50,progressBar=tqdm.tqdm_notebook,
              numSteps=float('inf')):
        for epoch in range(self.startEpoch,self.startEpoch+numEpochs):
            print("EPOCH",epoch)
            epochLoss = []
            step = 0
            for batch in progressBar(self.loader):
                if step >= numSteps: break
                step += 1
#                 bL,bS,BY,bI = [ch.autograd.Variable(t.to(self.dev)) for t in batch]
                batch = [ch.autograd.Variable(t.to(self.dev)) for t in batch]
#                 for i in (0,1):
#                     batch[i] = ch.autograd.Variable(batch[i].to(self.dev))
                loss = self.lossFun(self.network,batch)
                epochLoss.append(loss.data.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.lossHist.append(np.mean(epochLoss))
            self.startEpoch = len(self.lossHist)
            print('epoch',epoch,'total',self.lossHist[-1])
            self.bestLoss = min(self.lossHist[-1],self.bestLoss)
            self.dispFun(self.network,batchV)
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
        saveChkpt(state,model=self.modelName,params=self.params)
    
    def load(self,best=0):
        self.startEpoch,self.lossHist,self.bestLoss = loadChkpt(self.network,self.optimizer,self.modelName,self.dev,self.root,self.params,best)
        self.startEpoch = len(self.lossHist)
        
    def to(self,dev):
        self.dev = dev
        return self
    