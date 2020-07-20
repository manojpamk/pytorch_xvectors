#!/bin/python3.6

"""
    Date Created: Apr 6 2020

    Training script for relation networks

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

# SEEDS
torch.manual_seed(0)
np.random.seed(0)

# PARAMS, MODEL PREP
parser = getParams()
args = parser.parse_args()
checkParams(args)
print(args)

totalEpisodes = args.totalEpisodes
encoder_net, relation_net, encoder_optimizer, relation_optimizer, episodeI, saveDir = prepareRelationModel(args)
currLR = encoder_optimizer.param_groups[0]['lr']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numBatchesPerArk = int(args.numEgsPerArk/args.batchSize)

# LR SCHEDULERS
encoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(encoder_optimizer, gamma=0.95)
relation_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(relation_optimizer, gamma=0.95)
criterion = nn.MSELoss()
criterion_xent = nn.CrossEntropyLoss()

encoder_optimizer.param_groups[0]['lr'] = currLR
relation_optimizer.param_groups[0]['lr'] = currLR
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
        encoder_optimizer.zero_grad()
        relation_optimizer.zero_grad()
        episode_start_time = time.time()
        numClasses = int(len(x)/samplesPerClass)
        x = x.view(samplesPerClass, numClasses, -1, args.featDim)
        supports = x[:numSupports,:,:,:].detach()
        queries = x[numSupports:,:,:,:].detach()

        encoder_sup = encoder_net(
            supports.view(-1, supports.shape[2], args.featDim).permute(0,2,1).to(device), eps)
        encoder_quer = encoder_net(
            queries.view(-1, queries.shape[2], args.featDim).permute(0,2,1).to(device), eps)

        # Computing sum across supports within each class
        encoder_dim = encoder_sup.shape[-1]
        encoder_sup = torch.sum(encoder_sup.view(-1, numClasses, encoder_dim), dim=0)

        encoder_sup = encoder_sup.unsqueeze(0).expand(numClasses, numClasses, encoder_dim)
        encoder_quer = encoder_quer.unsqueeze(1).expand(numClasses, numClasses, encoder_dim)

        relation_out = relation_net(torch.cat((encoder_sup, encoder_quer), dim=2).view(-1, 2*encoder_dim))

        # X-ent loss
        labels = torch.arange(0,numClasses)
        loss = criterion_xent(relation_out.view(numClasses, numClasses), labels.cuda())        
        loggingLoss.append(loss.item())

        if np.isnan(loss.item()):
            print('Nan encountered at iter %d. Exiting..' %iter)
            sys.exit(1)
        loss.backward()
        encoder_optimizer.step()
        relation_optimizer.step()
        print('Episode time: %1.3f   Episode Loss: %1.3f'  %(time.time()-episode_start_time, loss.item()))
        episodeI += 1

    if episodeI%(10*args.protoEpisodesPerArk) == 0:
        encoder_lr_scheduler.step()
        relation_lr_scheduler.step()

    # Log, as long as episodeI <= totalEpisodes
    print('Episode: (%d/%d)     Avg Loss/batch: %1.3f' %(
        episodeI,
        totalEpisodes,
        np.mean(loggingLoss)))

    print('Archive time: %1.3f' %(time.time()-archive_start_time))

    # Save checkpoint
    torch.save({
        'episodeI': episodeI,
        'encoder_state_dict': encoder_net.state_dict(),
        'relation_state_dict': relation_net.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'relation_optimizer_state_dict': relation_optimizer.state_dict(),
        'args': args,
        }, '{}/checkpoint_episode_{}.tar'.format(saveDir, episodeI))

    if episodeI > 10*args.protoEpisodesPerArk:
        if os.path.exists('%s/checkpoint_step_%d.tar' %(saveDir,episodeI-10*args.protoEpisodesPerArk)):
            if episodeI%(50*args.protoEpisodesPerArk) !=0:
                os.remove('%s/checkpoint_step_%d.tar' %(saveDir,episodeI-10*args.protoEpisodesPerArk))
