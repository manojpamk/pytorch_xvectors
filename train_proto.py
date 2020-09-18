#!/bin/python3.6

"""
    Date Created: Apr 6 2020

    Training script for prototypical networks

"""

import os
import sys
import glob
import time
import socket
import torch
import numpy as np
from train_utils import *
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

def euclideanLoss(embed_quer, prototypes):
    """
        prototypes: (N, D)
        embed_quer: (M, N, D)

        D: embedding dimension
        N: number of classes
        M: samples per class

    """
    M, N, D = embed_quer.shape
    embed_quer = embed_quer.unsqueeze(2).expand(-1, -1, N, -1)
    prototypes = prototypes.view(1, 1, N, D).expand(M, N, -1, -1)
    logits = ((embed_quer - prototypes)**2).sum(dim=3)
    return -logits

# SEEDS
torch.manual_seed(0)
np.random.seed(0)

# PARAMS, MODEL PREP
parser = getParams()
args = parser.parse_args()
checkParams(args)
print(args)

totalEpisodes = args.totalEpisodes
net, optimizer, episodeI, saveDir = prepareProtoModel(args)
currLR = optimizer.param_groups[0]['lr']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numBatchesPerArk = int(args.numEgsPerArk/args.batchSize)

# LR SCHEDULERS
cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                          max_lr=args.maxLR,
                          cycle_momentum=False,
                          div_factor=5,
                          final_div_factor=1e+3,
                          total_steps=totalEpisodes,
                          pct_start=0.15)
exponential_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                          gamma=0.95)
criterion = nn.CrossEntropyLoss()
optimizer.param_groups[0]['lr'] = currLR
eps = args.noiseEps
featDir = args.featDir

# TRAINING
while episodeI < totalEpisodes:

    hdf5File = np.random.choice(glob.glob(featDir+'/*.hdf5'))
    print('Reading from archive %s' %os.path.basename(hdf5File))
    dataSet = myH5DL(hdf5File)
    samplesPerClass = np.random.randint(3,4)
    numSupports = samplesPerClass - 1
    numQueries = 1
    batchSampler = myH5DL_sampler(hdf5File,
                       minClasses=args.protoMinClasses,
                       maxClasses=args.protoMaxClasses,
                       samplesPerClass=samplesPerClass,
                       numEpisodes=args.protoEpisodesPerArk)
    dataLoader = DataLoader(dataSet, batch_sampler=batchSampler, num_workers=0)

    loggingLoss = []
    archive_start_time = time.time()
    for x, _ in dataLoader:
        optimizer.zero_grad()
        episode_start_time = time.time()
        numClasses = int(len(x)/samplesPerClass)
        x = x.view(samplesPerClass, numClasses, -1, args.featDim)
        supports = x[:numSupports,:,:,:].detach()
        queries = x[numSupports:,:,:,:].detach()
        labels = torch.arange(numClasses).repeat(numQueries)

        embed_sup = net(
            supports.view(-1, supports.shape[2], args.featDim).permute(0,2,1).to(device), eps)
        embed_quer = net(
            queries.view(-1, queries.shape[2], args.featDim).permute(0,2,1).to(device), eps)

        # Prototype computation
        prototypes = embed_sup.view(supports.shape[0], supports.shape[1], -1).mean(dim=0)

        # Euclidean-softmax
        logits = euclideanLoss(embed_quer.view(queries.shape[0], queries.shape[1], -1), prototypes)

        # Original implementation of loss function
        loss = criterion(logits.view(numQueries*numClasses,numClasses), labels.to(device))

        # print(loss.item())
        loggingLoss.append(loss.item())

        loss.backward()
        optimizer.step()
        print('Episode time: %1.3f   Episode Loss: %1.3f'  %(time.time()-episode_start_time, loss.item()))
        del x, supports, queries, embed_sup, embed_quer, loss, logits, prototypes
        episodeI += 1

    if episodeI%(10*args.protoEpisodesPerArk) == 0:
        exponential_lr_scheduler.step()
    # Log, as long as episodeI <= totalEpisodes
    print('Episode: (%d/%d)     Avg Loss/batch: %1.3f' %(
        episodeI,
        totalEpisodes,
        np.mean(loggingLoss)))

    print('Archive time: %1.3f' %(time.time()-archive_start_time))

    # Save checkpoint
    torch.save({
        'episodeI': episodeI,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        }, '{}/checkpoint_episode_{}.tar'.format(saveDir, episodeI))

    if episodeI > 10*args.protoEpisodesPerArk:
        if os.path.exists('%s/checkpoint_step_%d.tar' %(saveDir,episodeI-10*args.protoEpisodesPerArk)):
            if episodeI%(50*args.protoEpisodesPerArk) !=0:
                os.remove('%s/checkpoint_step_%d.tar' %(saveDir,episodeI-10*args.protoEpisodesPerArk))
