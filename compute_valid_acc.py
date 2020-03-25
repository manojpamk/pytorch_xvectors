#!/bin/python3.6

""" Date Created: Feb 17 2020
    This script computes the validation accuracy
"""

import os
import sys
import torch
import socket
import kaldi_python_io
from train_utils import *


egsDir =
modelDir = '/home/manoj/Projects/pytorch_spkembed/xvectors_voxceleb/models/isXvec_False_modelType_3_event_202002-1719-0729'
modelFile = max(glob.glob(modelDir), key=os.path.getctime)

# Load the model
net = simpleTDNN(params['numSpkrs'], p_dropout=0)
checkpoint = torch.load(modelFile)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

correct, incorrect = 0, 0
for validArk in glob.glob(egsDir+'/valid_egs.*.ark'):
    x = kaldi_python_io.Nnet3EgsReader(validArk)
    for key, mat in x:
        out = net(torch.Tensor(mat[0]['matrix']).permute(1,0).unsqueeze(0))
        if mat[1]['matrix'][0][0][0] == torch.argmax(out)+1:
            correct += 1
        else:
            incorrect += 1
        #print('%d,%d' %(mat[1]['matrix'][0][0][0],torch.argmax(out)+1))
print('Valid accuracy: %1.2f percent' %(1.0*correct/(correct+incorrect)))
