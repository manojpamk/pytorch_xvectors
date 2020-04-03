#!/bin/bash
: ' Date Created: Apr 2 2020

    Speaker verification on the voices corpora

    voices-eval:
    EER: 10.3%
    minDCF(p-target=0.01): 0.7845
    minDCF(p-target=0.001): 0.9406
'

currDir=$PWD
kaldiDir=/home/manoj/kaldi
expDir=$currDir/exp_voices_kaldi
wavDir=/home/manoj/Datasets/voices/Speaker_Recognition/sid_eval
trialsFile=/home/manoj/Datasets/voices/eval_trials
wavList=$currDir/wavList
readlink -f $wavDir/* > $wavList
dataDir=$expDir/data
featDir=$expDir/feats

for f in sid steps utils local conf diarization; do
  [ ! -L $f ] && ln -s $kaldiDir/egs/voxceleb/v2/$f;
done

. cmd.sh
. path.sh

# Kaldi data preparation
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

# Feature extraction pipeline 
steps/make_mfcc.sh --write-utt2num-frames true \
  --mfcc-config conf/mfcc.conf --nj 16 --cmd "$train_cmd" \
  $dataDir
utils/fix_data_dir.sh $dataDir
sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" $dataDir
utils/fix_data_dir.sh $dataDir

local/nnet3/xvector/prepare_feats_for_egs.sh --nj 8 --cmd "$train_cmd" \
  $dataDir $featDir $expDir/data_no_sil
utils/fix_data_dir.sh $featDir

# Kaldi xvectors
nnetDir=$kaldiDir/egs/voxceleb/v2/exp/xvector_nnet_1a
transformDir=$nnetDir/xvectors_train
sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 16 \
  $nnetDir $dataDir \
  $expDir/kaldi_xvectors

# Scoring
$train_cmd $expDir/log_scores.log \
ivector-plda-scoring --normalize-length=true \
"ivector-copy-plda --smoothing=0.0 $transformDir/plda - |" \
"ark:ivector-subtract-global-mean $transformDir/mean.vec scp:$expDir/kaldi_xvectors/xvector.scp ark:- | transform-vec $transformDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
"ark:ivector-subtract-global-mean $transformDir/mean.vec scp:$expDir/kaldi_xvectors/xvector.scp ark:- | transform-vec $transformDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
"cat '$trialsFile' | cut -d\  --fields=1,2 |" $expDir/scores_eval

eer=`compute-eer <(local/prepare_for_eer.py $trialsFile $expDir/scores_eval) 2> /dev/null`
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $expDir/scores_eval $trialsFile 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $expDir/scores_eval $trialsFile 2> /dev/null`
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
