#!/bin/bash

: ' Date Created: Apr 27 2019

    A wrapper script for the diarization evals. For every model checkpoint,
    this computes the oracle and est spkr DERs

'

currDir=$PWD

# 'xvecTDNN' or 'proto_xvecTDNN' or 'relation_encoder_xvecTDNN'
modelType=proto_xvecTDNN
# 'fc2' or 'fc3' or 'fc4'
layerName=fc3
modelDir=$currDir/../models/temp_eval/

# 'dihard' or 'ami' or 'adosMod3'
evalCorpus=dihard
wavDir=$currDir/${evalCorpus}_wav
rttmDir=$currDir/${evalCorpus}_rttm

# 'plda' or 'SC'
method=SC  

expDir=$currDir/exp
baseScript=$currDir/diarize.sh
collarCmd="--ignore_overlaps"
outFile=$currDir/RESULTS.txt

rm -f $outFile
sed -i "/^wavDir=/c\wavDir=$wavDir" $baseScript
sed -i "/^rttmDir=/c\rttmDir=$rttmDir" $baseScript
sed -i "/^modelDir=/c\modelDir=$modelDir" $baseScript
sed -i "/^expDir=/c\expDir=$expDir" $baseScript
sed -i "/^method=/c\method=$method" $baseScript
sed -i "/^modelType=/c\modelType=$modelType" $baseScript
sed -i "/^layerName=/c\layerName=$layerName" $baseScript

for modelFile in $modelDir/*.tar; do

  touch $modelFile;
  echo "Evaluating $modelFile"

  bash diarize.sh > /dev/null 2>&1

  # Eval is a repeatition from inside $baseScript, but its OK
  cd $currDir/dscore/
  oracleResults=`python score.py -R <(ls $rttmDir/*) \
    -S <(ls $expDir/$method/clustering_oracleNumSpkr/rttm) |\
    grep OVERALL | tr -s ' ' | cut -f 4-5 -d ' '`
  estResults=`python score.py -R <(ls $rttmDir/*) \
    -S <(ls $expDir/$method/clustering_estNumSpkr/rttm) |\
    grep OVERALL | tr -s ' ' | cut -f 4-5 -d ' '`
  cd ..
  echo "       WITHOUT COLLAR                "
  echo "`basename $modelFile` `echo $oracleResults | cut -f 1 -d ' '` `echo $estResults | cut -f 1 -d ' '`" >> $outFile

  cd $currDir/dscore/
  oracleResults=`python score.py $collarCmd -R <(ls $rttmDir/*) \
    -S <(ls $expDir/$method/clustering_oracleNumSpkr/rttm) |\
    grep OVERALL | tr -s ' ' | cut -f 4-5 -d ' '`
  estResults=`python score.py $collarCmd -R <(ls $rttmDir/*) \
    -S <(ls $expDir/$method/clustering_estNumSpkr/rttm) |\
    grep OVERALL | tr -s ' ' | cut -f 4-5 -d ' '`
  cd ..
  echo "       WITH COLLAR                "
  echo "`basename $modelFile` `echo $oracleResults | cut -f 1 -d ' '` `echo $estResults | cut -f 1 -d ' '`" >> $outFile


done
