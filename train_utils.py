#!/bin/python3.6

"""
    Date Created: Feb 11 2020
    This file will contain the training utils

"""

import os
import glob
import h5py
import torch
import configparser
import argparse
from datetime import datetime
import numpy as np
from models import *
import kaldi_python_io

from kaldiio import ReadHelper

from torch.utils.data import Dataset, IterableDataset
#from apex.parallel import DistributedDataParallel
from collections import OrderedDict


def readHdf5File_full(fileName):
    """ Read at-once from the hdf5 file. Rarely used
        Outputs:
        feats: (N,1,chunkLen,30)
        labels: (N,1)
    """
    with h5py.File(fileName,'r') as x:
        feats, labels = np.array(x.get('feats')), np.array(x.get('labels'))
    chunkLen = feats.shape[1]
    feats = torch.from_numpy(feats).unsqueeze(1) # make in (N,1,chunkLen,30)
    labels = torch.from_numpy(labels)
    return feats, labels

class nnet3EgsDL(IterableDataset):
    """ Data loader class to read directly from egs files, no HDF5
    """

    def __init__(self, arkFile):
        self.fid = kaldi_python_io.Nnet3EgsReader(arkFile)

    def __iter__(self):
        return iter(self.fid)


class myH5DL(Dataset):
    """ Data loader class customized to reading from hdf5 files
    """

    def __init__(self, hdf5File):
        x = h5py.File(hdf5File,'r')
        self.feats = x.get('feats')
        self.labels = x.get('labels')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return samples from idx:idx+batch_size
        """
        X = self.feats[idx,:,:]
        Y = self.labels[idx]
        return X, Y

def prepareModel(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.benchmark = True

    if args.resumeTraining:
        # select the latest model from modelDir
        modelFile = max(glob.glob(args.resumeModelDir+'/*'), key=os.path.getctime)
        net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))
        # if args.modelType == 3:
        #     net = simpleTDNN(args.numSpkrs, p_dropout=0)
        # else:
        #     net = xvecTDNN(args.numSpkrs, p_dropout=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.baseLR)
        net.to(device)

        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
            else:
                new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)

        step = checkpoint['step']
        totalSteps = args.numEpochs * args.numArchives
        print('Resuming training from step %d' %step)

        # set the dropout
        if 1.0*step < args.stepFrac*totalSteps:
            p_drop = args.pDropMax*step*args.stepFrac/totalSteps
        else:
            p_drop = max(0,args.pDropMax*(totalSteps + args.stepFrac - 2*step)/(totalSteps - totalSteps*args.stepFrac))
        for x in net.modules():
            if isinstance(x, torch.nn.Dropout):
                x.p = p_drop

        saveDir = args.resumeModelDir
    else:
        print('Initializing Model..')
        step = 0
        net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))
        # if args.modelType == 3:
        #     net = simpleTDNN(args.numSpkrs, p_dropout=0)
        # else:
        #     net = xvecTDNN(args.numSpkrs, p_dropout=0)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.baseLR)
        net.to(device)
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[0],
                                                        output_device=0)
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        eventID = datetime.now().strftime('%Y%m-%d%H-%M%S')
        saveDir = './models/modelType_{}_event_{}' .format(args.modelType, eventID)
        os.makedirs(saveDir)

    return net, optimizer, step, saveDir


def getParams():
    parser = argparse.ArgumentParser()

    # PyTorch distributed run
    parser.add_argument("--local_rank", type=int, default=0)

    # General Parameters
    parser.add_argument('-modelType', default='xvecTDNN', help='Model class. Check models.py')
    parser.add_argument('-featDim', default=30, type=int, help='Frame-level feature dimension')
    parser.add_argument('-resumeTraining', default=0, type=int,
        help='(1) Resume training, or (0) Train from scratch')
    parser.add_argument('-resumeModelDir', default='', help='Path containing training checkpoints')

    # Training Parameters - no more trainFullXvector = 0
    parser.add_argument('-numArchives', default=84, type=int, help='Number of egs.*.ark files')
    parser.add_argument('-numSpkrs', default=7323, type=int, help='Number of output labels')
    parser.add_argument('-logStepSize', default=200, type=int, help='Iterations per log')
    parser.add_argument('-batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('-numEgsPerArk', default=366150, type=int,
        help='Number of training examples per egs file')
    parser.add_argument('egsDir', help='Directory with training archives')

    # Optimization Params
    parser.add_argument('-preFetchRatio', default=30, type=int, help='xbatchSize to fetch from dataloader')
    parser.add_argument('-optimMomentum', default=0.5, type=float, help='Optimizer momentum')
    parser.add_argument('-baseLR', default=1e-3, type=float, help='Initial LR')
    parser.add_argument('-maxLR', default=2e-3, type=float, help='Maximum LR')
    parser.add_argument('-numEpochs', default=2, type=int, help='Number of training epochs')
    parser.add_argument('-noiseEps', default=1e-5, type=float, help='Noise strength before pooling')
    parser.add_argument('-pDropMax', default=0.2, type=float, help='Maximum dropout probability')
    parser.add_argument('-stepFrac', default=0.5, type=float,
        help='Training iteration when dropout = pDropMax')

    return parser


def computeValidAccuracy(args, modelDir):
    """ Computes frame-level validation accruacy
    """
    modelFile = max(glob.glob(modelDir+'/*'), key=os.path.getctime)
    # Load the model
    net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))
    # if args.modelType == 3:
    #    net = simpleTDNN(args.numSpkrs, p_dropout=0)
    # else:
    #     net = xvecTDNN(args.numSpkrs, p_dropout=0)

    checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            new_state_dict[k] = v
    # load params
    net.load_state_dict(new_state_dict)
    net = net.cuda()
    net.eval()

    correct, incorrect = 0, 0
    for validArk in glob.glob(args.egsDir+'/valid_egs.*.ark'):
        x = kaldi_python_io.Nnet3EgsReader(validArk)
        for key, mat in x:
            out = net(x=torch.Tensor(mat[0]['matrix']).permute(1,0).unsqueeze(0).cuda(),eps=0)
            if mat[1]['matrix'][0][0][0]+1 == torch.argmax(out)+1:
                correct += 1
            else:
                incorrect += 1
    return 100.0*correct/(correct+incorrect)


def par_core_extractXvectors(inFeatsScp, outXvecArk, outXvecScp, net, layerName):
    """ To be called using pytorch multiprocessing
        Note: This function reads all the data from feats.scp into memory
        before inference. Hence, make sure the file is not too big (Hint: use
        split_data_dir.sh)
    """

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    eval('net.%s.register_forward_hook(get_activation(layerName))' %layerName)

    with kaldi_python_io.ArchiveWriter(outXvecArk, outXvecScp, matrix=False) as writer:
        with ReadHelper('scp:%s'%inFeatsScp) as reader:
            for key, mat in reader:
                out = net(x=torch.Tensor(mat).permute(1,0).unsqueeze(0).cuda(),
                          eps=0)
                writer.write(key, np.squeeze(activation[layerName].cpu().numpy()))
