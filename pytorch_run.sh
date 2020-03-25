#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2020   Manoj Kumar
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh

voxcelebDir=/home/manoj/kaldi/egs/voxceleb/v2/

mfccdir=$voxcelebDir/mfcc
vaddir=$voxcelebDir/mfcc

# The trials file is downloaded by local/make_voxceleb1_v2.pl.
voxceleb1_trials=$voxcelebDir/data/voxceleb1_test/trials
voxceleb1_root=$voxcelebDir/vox_data/vox1
voxceleb2_root=$voxcelebDir/vox_data/vox2
musan_root=/home/manoj/Datasets/musan

stage=1

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  # This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
  # if you downloaded the dataset soon after it was released, you will want to use the make_voxceleb1.pl script instead.
  # local/make_voxceleb1.pl $voxceleb1_root data
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,323 speakers and 1,276,888 utterances.
  utils/combine_data.sh $PWD/data/train $PWD/data/voxceleb2_train $PWD/data/voxceleb2_test $PWD/data/voxceleb1_train
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      $PWD/data/${name} $PWD/exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $PWD/data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      $PWD/data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh $PWD/data/${name}
  done
fi

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi


if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
  utils/fix_data_dir.sh data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/train_aug_1m exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
  utils/fix_data_dir.sh data/train_combined_no_sil

  # Preparing the test features as well. This will be used only during testing
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 10 --cmd "$train_cmd" \
    data/voxceleb1_test data/voxceleb1_test_no_sil exp/voxceleb1_test_no_sil
  utils/fix_data_dir.sh data/voxceleb1_test_no_sil


fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
  mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
  mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk

  utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
  mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_combined_no_sil
fi

exit 0
# Run the following script stage-by-stage for the first time

# Main DNN training script yet to be integrated


# Xvector extraction
configFile=elessar.config.inference
extractModel=`grep ^extractModel $configFile | cut -f 3 -d ' '`
trainFeatDir=`grep ^trainFeatDir $configFile | cut -f 3 -d ' '`
testFeatDir=`grep ^testFeatDir $configFile | cut -f 3 -d ' '`
trainXvecDir=xvectors/$extractModel/train/
testXvecDir=xvectors/$extractModel/test/

time python3 extract.py elessar.config
cat $trainXvecDir/xvector.*.scp > $trainXvecDir/xvector.scp
cat $testXvecDir/xvector.*.scp > $testXvecDir/xvector.scp


# Compute the mean vector for centering the evaluation xvectors.
$train_cmd $trainXvecDir/log/compute_mean.log \
  ivector-mean scp:$trainXvecDir/xvector.scp \
  $trainXvecDir/mean.vec

# This script uses LDA to decrease the dimensionality prior to PLDA.
lda_dim=200
$train_cmd $trainXvecDir/log/lda.log \
  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
  "ark:ivector-subtract-global-mean scp:$trainXvecDir/xvector.scp ark:- |" \
  ark:$trainFeatDir/utt2spk $trainXvecDir/transform.mat

# Train the PLDA model.
$train_cmd $trainXvecDir/log/plda.log \
  ivector-compute-plda ark:$trainFeatDir/spk2utt \
  "ark:ivector-subtract-global-mean scp:$trainXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
  $trainXvecDir/plda

$train_cmd $testXvecDir/log/voxceleb1_test_scoring.log \
  ivector-plda-scoring --normalize-length=true \
  "ivector-copy-plda --smoothing=0.0 $trainXvecDir/plda - |" \
  "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $testXvecDir/scores_voxceleb1_test

eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $testXvecDir/scores_voxceleb1_test) 2> /dev/null`
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $testXvecDir/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $testXvecDir/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
### Kaldi x-vectors
# EER: 3.128%
# minDCF(p-target=0.01): 0.3258
# minDCF(p-target=0.001): 0.5003

