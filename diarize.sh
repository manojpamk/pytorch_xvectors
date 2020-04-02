#!/bin/bash

: ' Date Created: Mar 30 2020
    Perform speaker diarization on a provided corpus
'


currDir=$PWD
kaldiDir=/home/manoj/kaldi
expDir=$currDir/exp
wavDir=$currDir/demo_wav
rttmDir=$currDir/demo_rttm
wavList=$currDir/wavList
readlink -f $wavDir/* > $wavList

# Extraction parameters
window=1.5
window_period=0.75
min_segment=0.5

# Evaluation parameters
useOracleNumSpkr=1
useCollar=1
skipDataPrep=0
dataDir=$expDir/data
nj=8

for f in sid steps utils local conf diarization; do
  [ ! -L $f ] && ln -s $kaldiDir/egs/voxceleb/v2/$f;
done

if [[ "$useCollar" == "1" ]]; then
    collarCmd="-1 -c 0.25"
else
    collarCmd=""
fi

. cmd.sh
. path.sh

# Kaldi directory preparation
if [ "$skipDataPrep" == "0" ]; then

  rm -rf $dataDir; mkdir -p $dataDir
  paste -d ' ' <(rev $wavList | cut -f 1 -d '/' | rev | sed "s/\.wav$/-rec/g") \
    <(cat $wavList | xargs readlink -f) > $dataDir/wav.scp
  paste -d ' ' <(cut -f 1 -d ' ' $dataDir/wav.scp | sed "s/-rec$//g") \
    <(cut -f 1 -d ' ' $dataDir/wav.scp | sed "s/-rec$//g") > $dataDir/utt2spk
  cp $dataDir/utt2spk $dataDir/spk2utt
  numUtts=`wc -l $dataDir/utt2spk | cut -f 1 -d ' '`
  paste -d ' ' <(cut -f 1 -d ' ' $dataDir/utt2spk) \
    <(cut -f 1 -d ' ' $dataDir/wav.scp) <(yes "0" | head -n $numUtts) <(cat $wavList | xargs soxi -D) \
    >  $dataDir/segments
  if [ "$useOracleNumSpkr" == "1" ]; then
    for rttmFile in $rttmDir/*.rttm; do
      n=`cut -f 8 -d ' ' $rttmFile | sort | uniq | wc -l`
      echo "`basename $rttmFile .rttm` $n" >> $dataDir/reco2num_spk
    done
  fi

  # Create VAD directory
  if [ ! -d $currDir/oracleVAD ] || [ ! -d $currDir/evalVAD ]; then
    python convert_rttm_to_vad.py $wavDir $rttmDir $currDir/oracleVAD
  fi

  while read -r line; do
      uttID=`echo $line | cut -f 1 -d ' '`
      inVadFile=$currDir/oracleVAD/$uttID.csv
      [ ! -f $inVadFile ] && { echo "Input vad file does not exist"; exit 0; }
      paste -d ' ' <(echo $uttID) <(cut -f 2 -d ',' $inVadFile | tr "\n" " " | sed "s/^/ [ /g" | sed "s/$/ ]/g") >> $dataDir/vad.txt
  done < $dataDir/utt2spk
  copy-vector ark,t:$dataDir/vad.txt ark,scp:$dataDir/vad.ark,$dataDir/vad.scp

  # Feature preparation pipeline, until train_combined_no_sil
  utils/fix_data_dir.sh $dataDir
  steps/make_mfcc.sh --nj $nj \
                 --cmd "$train_cmd" \
                 --mfcc-config conf/mfcc.conf \
                 --write-utt2num-frames true \
                 $dataDir || exit 1
  utils/fix_data_dir.sh $dataDir

  diarization/vad_to_segments.sh --nj $nj \
                 --cmd "$train_cmd" \
                 --segmentation-opts '--silence-proportion 0.01001' \
                 --min-duration 0.5 \
                 $dataDir $dataDir/segmented || exit 1

  local/nnet3/xvector/prepare_feats.sh --nj $nj \
                 --cmd "$train_cmd" \
                 $dataDir/segmented \
                 $dataDir/segmented_cmn \
                 $dataDir/segmented_cmn/feats || exit 1
  cp $dataDir/segmented/segments $dataDir/segmented_cmn/segments
  utils/fix_data_dir.sh $dataDir/segmented_cmn
  utils/split_data.sh $dataDir/segmented_cmn $nj
else
  [ ! -d $dataDir/segmented_cmn ] && echo "Cannot find features" && exit 1
fi

# Compute the subsegments directory
utils/data/get_uniform_subsegments.py \
             --max-segment-duration=$window \
             --overlap-duration=$(perl -e "print ($window-$window_period);") \
             --max-remaining-duration=$min_segment \
             --constant-duration=True \
            $dataDir/segmented_cmn/segments > $dataDir/segmented_cmn/subsegments
utils/data/subsegment_data_dir.sh $dataDir/segmented_cmn \
  $dataDir/segmented_cmn/subsegments $dataDir/pytorch_xvectors/subsegments
utils/split_data.sh $dataDir/pytorch_xvectors/subsegments $nj

# Extract x-vectors
modelDir=models/xvec_preTrained
transformDir=xvectors/xvec_preTrained/train

python extract.py -modelDirectory $modelDir \
  -featDir $dataDir/pytorch_xvectors/subsegments \
  -embeddingDir $dataDir/pytorch_xvectors

for f in segments utt2spk spk2utt; do
  cp $dataDir/pytorch_xvectors/subsegments/$f $dataDir/pytorch_xvectors/$f
done

diarization/nnet3/xvector/score_plda.sh --nj 16 \
             --cmd "$train_cmd" \
             $transformDir \
             $dataDir/pytorch_xvectors \
             $dataDir/pytorch_xvectors/scoring

diarization/cluster.sh --nj 8 \
             --cmd "$train_cmd --mem 5G" \
             --reco2num-spk $dataDir/reco2num_spk \
             $dataDir/pytorch_xvectors/scoring \
             $dataDir/pytorch_xvectors/clustering_oracleNumSpkr

diarization/cluster.sh --nj 8 \
             --cmd "$train_cmd --mem 5G" \
             --threshold 0 \
             $dataDir/pytorch_xvectors/scoring \
             $dataDir/pytorch_xvectors/clustering_estNumSpkr

# Evaluation
echo "DER with Oracle #Spkrs"
perl md-eval.py $collarCmd -r <(cat $rttmDir/*) \
  -s <(sed "s/-rec / /g" $dataDir/pytorch_xvectors/clustering_oracleNumSpkr) \
  2>&1 | grep -v WARNING | grep OVERALL

echo "DER with Estimated #Spkrs"
perl md-eval.py $collarCmd -r <(cat $rttmDir/*) \
  -s <(sed "s/-rec / /g" $dataDir/pytorch_xvectors/clustering_estNumSpkr) \
  2>&1 | grep -v WARNING | grep OVERALL

rm $wavList
