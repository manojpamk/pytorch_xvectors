#!/bin/python

"""
    Date Created: Feb 26 2018

    This script extracts trained embeddings given the model directory, and saves them in kaldi format

"""
import os
import sys
import glob
import argparse
import kaldi_io
from models import *
import kaldi_python_io
import socket
from train_utils import *
from collections import OrderedDict
from torch.multiprocessing import Pool, Process, set_start_method
torch.multiprocessing.set_start_method('spawn', force=True)

def getSplitNum(text):
    return int(text.split('/')[-1].lstrip('split'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelType', default=4, type=int, help='Refer train_utils.py ')
    parser.add_argument('-numSpkrs', default=7323, type=int, help='Number of output labels for model')
    parser.add_argument('modelDirectory', help='Directory containing the model checkpoints')
    parser.add_argument('featDir', help='Directory containing features ready for extraction')
    parser.add_argument('embeddingDir', help='Output directory')
    args = parser.parse_args()

    modelFile = max(glob.glob(args.modelDirectory+'/*'), key=os.path.getctime)
    # Load model definition
    if args.modelType == 3:
        net = simpleTDNN(args.numSpkrs, p_dropout=0)
    else:
        net = xvecTDNN(args.numSpkrs, p_dropout=0)

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

    # Parallel Processing
    try:
        nSplits = int(sorted(glob.glob(args.featDir+'/split*'),
                  key=getSplitNum)[-1].split('/')[-1].lstrip('split'))
    except:
        print('Cannot find %s/splitN directory' %args.featDir)
        sys.exit(1)

    if not os.path.isdir(args.embeddingDir):
        os.makedirs(args.embeddingDir)
    nProcs = nSplits
    L = [('%s/split%d/%d/feats.scp' %(args.featDir, nSplits, i),
        '%s/xvector.%d.ark' %(args.embeddingDir, i),
        '%s/xvector.%d.scp' %(args.embeddingDir, i), net, 'fc1' ) for i in range(1,nSplits+1)]
    pool2 = Pool(processes=nProcs)
    result = pool2.starmap(par_core_extractXvectors, L )
    pool2.terminate()

    os.system('cat %s/xvector.*.scp > %s/xvector.scp' %(args.embeddingDir, args.embeddingDir))

if __name__ == "__main__":
    main()
