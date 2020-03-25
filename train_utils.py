#!/bin/python3.6

"""
    Date Created: Feb 11 2020
    This file will contain the training utils

"""

import os
import glob
import h5py
import torch
import kaldi_io
import configparser
from datetime import datetime
import numpy as np
from models import *
import kaldi_python_io
from torch.utils.data import Dataset, IterableDataset
#from apex.parallel import DistributedDataParallel
from collections import OrderedDict


def selectModel(modelType):
    if modelType == 1:
        net = simpleCNN()
    if modelType == 2:
        net = simpleLSTM()
    if modelType == 3:
        net = simpleTDNN()
    if modelType == 4:
        net = xvecTDNN()
    return net

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

def prepareModel(params):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.benchmark = True

    if params['resumeTraining']:
        # select the latest model from modelDir
        modelFile = max(glob.glob(params['resumeModelDir']+'/*'), key=os.path.getctime)
        if params['modelType'] == 3:
            net = simpleTDNN(params['numSpkrs'], p_dropout=0)
        else:
            net = xvecTDNN(params['numSpkrs'], p_dropout=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=params['base_lr'])
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
        totalSteps = params['numEpochs'] * params['numArchives']
        print('Resuming training from step %d' %step)

        # set the dropout
        if 1.0*step < params['step_frac']*totalSteps:
            p_drop = params['p_drop_max']*step*params['step_frac']/totalSteps
        else:
            p_drop = max(0,params['p_drop_max']*(totalSteps + params['step_frac'] - 2*step)/(totalSteps - totalSteps*params['step_frac']))
        for x in net.modules():
            if isinstance(x, torch.nn.Dropout):
                x.p = p_drop

        saveDir = params['resumeModelDir']
    else:
        print('Initializing Model..')
        step = 0
        if params['modelType'] == 3:
            net = simpleTDNN(params['numSpkrs'], p_dropout=0)
        else:
            net = xvecTDNN(params['numSpkrs'], p_dropout=0)

        optimizer = torch.optim.Adam(net.parameters(), lr=params['base_lr'])
        net.to(device)
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[0],
                                                        output_device=0)
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        eventID = datetime.now().strftime('%Y%m-%d%H-%M%S')
        saveDir = './models/isXvec_{}_modelType_{}_event_{}' .format(
                      params['trainFullXvector'], params['modelType'], eventID)
        os.mkdir(saveDir)

    return net, optimizer, step, saveDir


def getParams(configFile):

    paramDict = {}
    config = configparser.ConfigParser()
    config.read(configFile)
    paramDict['modelType'] = int(config['General']['modelType'])
    paramDict['featDim'] = int(config['General']['featDim'])
    paramDict['trainFullXvector'] = config['General'].getboolean('trainFullXvector')
    paramDict['resumeTraining'] = config['General'].getboolean('resumeTraining')
    paramDict['resumeModelDir'] = config['General']['resumeModelDir']

    if paramDict['trainFullXvector']:
        paramDict['numArchives'] = int(config['fullXvector']['numArchives'])
        paramDict['numSpkrs'] = int(config['fullXvector']['numSpkrs'])
        paramDict['logStepSize'] = int(config['fullXvector']['logStepSize'])
        paramDict['batchSize'] = int(config['fullXvector']['batchSize'])
        paramDict['numEgsPerArk'] = int(config['fullXvector']['numEgsPerArk'])
        paramDict['egsDir'] = config['fullXvector']['egsDir']
    else:
        paramDict['numArchives'] = int(config['toyXvector']['numArchives'])
        paramDict['numSpkrs'] = int(config['toyXvector']['numSpkrs'])
        paramDict['logStepSize'] = int(config['toyXvector']['logStepSize'])
        paramDict['batchSize'] = int(config['toyXvector']['batchSize'])
        paramDict['numEgsPerArk'] = int(config['toyXvector']['numEgsPerArk'])
        paramDict['egsDir'] = config['toyXvector']['egsDir']

    # Training params
    paramDict['preFetchRatio'] = int(config['Training']['preFetchRatio'])
    paramDict['optim_momentum'] = float(config['Optimizer']['momentum'])
    paramDict['base_lr'] = float(config['Optimizer']['base_lr'])
    paramDict['max_lr'] = float(config['Optimizer']['max_lr'])
    paramDict['numEpochs'] = int(config['Training']['numEpochs'])
    paramDict['noise_eps'] = float(config['Training']['noise_eps'])
    paramDict['p_drop_max'] = float(config['Training']['p_drop_max'])
    paramDict['step_frac'] = float(config['Training']['step_frac'])

    # Extraction params
    paramDict['extractModelName'] = config['Extraction']['extractModel']
    paramDict['extractModelDir'] = 'final_models/'+paramDict['extractModelName']
    paramDict['trainFeatDir'] = config['Extraction']['trainFeatDir']
    paramDict['testFeatDir'] = config['Extraction']['testFeatDir']
    paramDict['trainXvecDir'] = 'xvectors/{}/train'.format(paramDict['extractModelName'])
    paramDict['testXvecDir'] = 'xvectors/{}/test'.format(paramDict['extractModelName'])

    return paramDict



def computeValidAccuracy(params, modelDir):
    """ Computes frame-level validation accruacy
    """
    modelFile = max(glob.glob(modelDir+'/*'), key=os.path.getctime)
    # Load the model
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
    # load params
    net.load_state_dict(new_state_dict)
    net = net.cuda()
    net.eval()

    correct, incorrect = 0, 0
    for validArk in glob.glob(params['egsDir']+'/valid_egs.*.ark'):
        x = kaldi_python_io.Nnet3EgsReader(validArk)
        for key, mat in x:
            out = net(x=torch.Tensor(mat[0]['matrix']).permute(1,0).unsqueeze(0).cuda(),eps=0)
            if mat[1]['matrix'][0][0][0]+1 == torch.argmax(out)+1:
                correct += 1
            else:
                incorrect += 1
    return 100.0*correct/(correct+incorrect)



def par_core_extractXvectors(inFeatsScp, outXvecArk, outXvecScp, net):
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
    net.fc1.register_forward_hook(get_activation('fc1'))
    
    with kaldi_python_io.ArchiveWriter(outXvecArk, outXvecScp, matrix=False) as writer:
        for key, mat in kaldi_io.read_mat_scp(inFeatsScp):
            out = net(x=torch.Tensor(mat).permute(1,0).unsqueeze(0).cuda(),
                      eps=0)
            writer.write(key, np.squeeze(activation['fc1'].cpu().numpy()))




