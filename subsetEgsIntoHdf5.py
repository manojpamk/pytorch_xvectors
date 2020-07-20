#!/bin/python

""" Date Created: Apr 8 2020

    This script breaks down an nnet3-egs ark file into multiple hdf5 files
    suitable for protonet training
    [ Looking for a more direct way for this (or) Bypassing HDF5 for protonet ]

"""

import os
import sys
import glob
import h5py
import random
import subprocess
import numpy as np
import kaldi_python_io
from multiprocessing import Pool

def writeHdf5File(egsFile, scpFile, chunkLen, hdf5File):

    featDim = 30
    output = subprocess.run(['wc','-l',scpFile], stdout=subprocess.PIPE).stdout.decode('utf-8')
    numSamples = int(output.split()[0])
    x = kaldi_python_io.Nnet3EgsReader(egsFile)
    with h5py.File(hdf5File,'w') as fid:
        feats = fid.create_dataset('feats',(numSamples,chunkLen,featDim), dtype='f')#, compression="gzip")
        labels = fid.create_dataset('labels',(numSamples,1), dtype='i8')#, compression="gzip")
        count = 0
        for key,mat in x:
            labels[count] = mat[1]['matrix'][0][0][0]
            feats[count] = mat[0]['matrix']
            count += 1

if len(sys.argv) != 3:
    print('Usage: python subsetEgsIntoHdf5.py <egsDir> <hdf5Dir>')
    sys.exit(1)

egsDir = sys.argv[1]
hdf5Dir = sys.argv[2]
tempDir = hdf5Dir + '/temp/'
numSplits = 8

os.system('rm -rf %s' %hdf5Dir)
os.system('mkdir -p %s' %tempDir)

arkCount = 0
for fileI, scpFile in enumerate(sorted(glob.glob(egsDir+'/egs.*.scp'))):
   
    print('working on archive %d' %(arkCount+1))
    #if arkCount == 45:
    #    break
    arkCount += 1
    inputArkNum = int(os.path.basename(scpFile).split('.')[1])
    # First, read all the labels alongwith indices
    with open(scpFile,'r') as fid:
        data = fid.read().splitlines()
    chunkLen = int(data[0].split()[0].split('-')[-2])

    # Divide the speakers into splits with approx. equal speakers
    print('Creating new speaker lists..')
    spkrLoc = {}
    for i,x in enumerate(data):
        spkrID = int(x.split()[0].split('-')[-1])
        if spkrID in spkrLoc:
            spkrLoc[spkrID].append(i)
        else:
            spkrLoc[spkrID] = [i]
    uniqSpkrs = np.fromiter(spkrLoc.keys(), dtype=int)
    spkrSplits = [uniqSpkrs[i::numSplits] for i in range(numSplits)]
    for splitI,split in enumerate(spkrSplits):
        with open(tempDir+'/temp.{}.scp'.format(splitI+1),'w') as fid:
            for spkr in split:
                for loc in spkrLoc[spkr]:
                    fid.write('%s\n' %data[loc])

    print('Creating temporary egs files')
    copyCommands = [ 'nnet3-copy-egs scp:%s ark:%s > /dev/null 2>&1' %(
        tempDir+'/temp.{}.scp'.format(splitI+1),
        tempDir+'/temp.{}.ark'.format(splitI+1)) for splitI in range(numSplits)]

    # start all programs
    processes = [subprocess.Popen(program, shell=True) for program in copyCommands]
    # wait
    for process in processes:
        process.wait()

    print('Creating hdf5 files now..')
    nProcs = 8
    L = [(tempDir+'/temp.{}.ark'.format(splitI+1),
        tempDir+'/temp.{}.scp'.format(splitI+1),
        chunkLen,
        hdf5Dir+'/egs.{}.{}.hdf5'.format(inputArkNum,splitI+1)) for splitI in range(numSplits)]
    pool = Pool(processes=nProcs)
    results = pool.starmap(writeHdf5File, L)
    pool.terminate()

    # Cleanup
    os.system('rm -f %s/*' %tempDir)

os.system('rm -rf %s' %tempDir)
