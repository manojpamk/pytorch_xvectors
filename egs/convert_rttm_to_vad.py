#!/bin/python

"""
	This script converts the RTTM ground truth files into oracleVAD files to be
	used for computing SER (speaker error rate)

	Output format (xxx.csv):
	<time>,<label>

"""

import os, sys
import numpy as np

if len(sys.argv)!=4:
	print("Usage: convert_rttm_to_vad.py <audio_dir> <in_rttm_dir> <out_oracleVad_dir>\n")
	print("Converts RTTM ground truth files into oracleVAD files")
	sys.exit(1)

inWavDir = sys.argv[1]
inRttmDir = sys.argv[2]
outOracleVadDir = sys.argv[3]
frameRate = 100

if not os.path.exists(outOracleVadDir):
    os.makedirs(outOracleVadDir)

for wavFile in sorted(os.listdir(inWavDir)):

	wavBase = wavFile.replace('.wav','')
	if not os.path.exists(inRttmDir+'/'+wavBase+'.rttm'):
		print('No rttm file for %s' %wavFile)
	# print('Creating VAD file for %s' %wavBase)
	audioDur = np.round(float(os.popen('soxi -D '+inWavDir+'/'+wavFile).readlines()[0].strip('\n')),2)
	binVad = np.zeros(int(np.ceil(frameRate*audioDur))).astype('int')

	with open(inRttmDir+'/'+wavBase+'.rttm') as fid:
		data = fid.read().splitlines()
		startTimes = [ float(x.split()[3]) for x in data ]
		endTimes  = [ float(x.split()[3]) + float(x.split()[4]) for x in data ]

	for s,e in zip(startTimes, endTimes):
		binVad[int(np.ceil(frameRate*s)):int(np.ceil(frameRate*e))] = 1

	timeStamps = np.linspace(0.01,len(binVad)/100.0,len(binVad),endpoint=False)
	np.savetxt(outOracleVadDir+'/'+wavBase+'.csv',np.vstack((timeStamps,binVad)).T,fmt="%1.2f,%1.2f")
