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
    parser.add_argument('-modelType', default='xvecTDNN', help='Refer train_utils.py ')
    parser.add_argument('-numSpkrs', default=7323, type=int, help='Number of output labels for model')
    parser.add_argument('-layerName', default='fc1', help="DNN layer for embeddings")
    parser.add_argument('-nProcs', default=0, type=int, help='Number of parallel processes. Default=0(Number of input directory splits)')
    parser.add_argument('modelDirectory', help='Directory containing the model checkpoints')
    parser.add_argument('featDir', help='Directory containing features ready for extraction')
    parser.add_argument('embeddingDir', help='Output directory')
    args = parser.parse_args()

    # Checking for input features and splitN directories
    try:
        nSplits = int(sorted(glob.glob(args.featDir+'/split*'),
                  key=getSplitNum)[-1].split('/')[-1].lstrip('split'))
    except ValueError:
        print('[ERROR] Cannot find %s/splitN directory' %args.featDir)
        print('Use utils/split_data.sh to create this directory')
        sys.exit(1)

    if not os.path.isfile('%s/split%d/1/feats.scp' %(args.featDir, nSplits)):
        print('Cannot find input features')
        sys.exit(1)

    # Check for trained model
    try:
        modelFile = max(glob.glob(args.modelDirectory+'/*.tar'), key=os.path.getctime)
    except ValueError:
        print("[ERROR] No trained model has been found in {}.".format(args.modelDirectory) )
        sys.exit(1)

    # Load model definition
    net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))

    checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    if 'relation' in args.modelType:
        checkpoint_dict = checkpoint['encoder_state_dict']
    else:
        checkpoint_dict = checkpoint['model_state_dict']
    for k, v in checkpoint_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            new_state_dict[k] = v

    # load trained weights
    net.load_state_dict(new_state_dict)
    net = net.cuda()
    net.eval()

    if not os.path.isdir(args.embeddingDir):
        os.makedirs(args.embeddingDir)

    print('Extracting xvectors by distributing jobs to pool workers... ')
    if not args.nProcs:
        args.nProcs = nSplits

    L = [('%s/split%d/%d/feats.scp' %(args.featDir, nSplits, i),
        '%s/xvector.%d.ark' %(args.embeddingDir, i),
        '%s/xvector.%d.scp' %(args.embeddingDir, i), net, args.layerName ) for i in range(1,nSplits+1)]
    pool2 = Pool(processes=args.nProcs)
    result = pool2.starmap(par_core_extractXvectors, L )
    pool2.terminate()
    print('Multithread job has been finished.')

    print('Writing xvectors to {}'.format(args.embeddingDir))
    os.system('cat %s/xvector.*.scp > %s/xvector.scp' %(args.embeddingDir, args.embeddingDir))


if __name__ == "__main__":
    main()
