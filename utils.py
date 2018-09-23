import torch as ch
import os
from hyperparams import Hyperparams as params
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
paramDict = {'sep':params.sep,'alpha':params.alpha,
             'dropout':params.dropout,'norm':params.norm,
             'lr':params.lr,'chunk':params.chunk}

def saveChkpt(state,model,root='.'):
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


def loadChkpt(network,optimizer,model,dev='cpu',root='.'):
    chkptDirNm = "|".join("{}:{}".format(k,v) for k,v in paramDict.items())
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