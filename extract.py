#!/bin/python

"""
    Date Created: Feb 26 2018

    This script extracts trained embeddings given the model directory, and saves them in kaldi format

"""
import os
import sys
import glob
import kaldi_io
from models import *
import kaldi_python_io
import socket
from train_utils import *
from collections import OrderedDict
from torch.multiprocessing import Pool, Process, set_start_method
torch.multiprocessing.set_start_method('spawn', force=True)

def getSplitNum(text):
    return int(text.split('/')[-1].lstrip('split')

def main():

    if len(sys.argv) != 2:
        print('Usage: python3 extract.py configFile')
        sys.exit(0)

    params = getParams(sys.argv[1])
    modelFile = max(glob.glob(params['extractModelDir']+'/*'), key=os.path.getctime)
    # Load model definition
    if params['modelType'] == 3:
        net = simpleTDNN(params['numSpkrs'], p_dropout=0)
    else:
        net = xvecTDNN(params['numSpkrs'], p_dropout=0)

    checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            new_state_dict[k] = v

    # load trained weights
    net.load_state_dict(new_state_dict)
    net = net.cuda()
    net.eval()

    if not os.path.isdir(params['trainXvecDir']):
        os.makedirs(params['trainXvecDir'])
    if not os.path.isdir(params['testXvecDir']):
        os.makedirs(params['testXvecDir'])

    # Parallel Processing
    nProcs = 8

    # Identify the maximum splitN/ directory
    nSplits = int(sorted(glob.glob(params['testFeatDir']+'/split*'), key=getSplitNum)[-1].split('/')[-1].lstrip('split'))
    L = [('%s/split%d/%d/feats.scp' %(params['testFeatDir'], nSplits, i),
            '%s/xvector.%d.ark' %(params['testXvecDir'], i),
            '%s/xvector.%d.scp' %(params['testXvecDir'], i), net, 'fc1' ) for i in range(1,nSplits+1)]
    pool = Pool(processes=nProcs)
    result = pool.starmap(par_core_extractXvectors, L )
    pool.terminate()

    nSplits = int(sorted(glob.glob(params['trainFeatDir']+'/split*'), key=getSplitNum)[-1].split('/')[-1].lstrip('split'))
    L = [('%s/split%d/%d/feats.scp' %(params['trainFeatDir'], nSplits, i),
            '%s/xvector.%d.ark' %(params['trainXvecDir'], i),
            '%s/xvector.%d.scp' %(params['trainXvecDir'], i), net, 'fc1' ) for i in range(1,nSplits+1)]
    pool2 = Pool(processes=nProcs)
    result = pool2.starmap(par_core_extractXvectors, L )
    pool2.terminate()

if __name__ == "__main__":
    main()
