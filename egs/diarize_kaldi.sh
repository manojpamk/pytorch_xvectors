#!/bin/bash

: ' Date Created: Mar 27 2020
    Perform speaker diarization using kaldi xvectors

    All numbers in %meanDER: (oracle #spkr, estimated #spkr)
    DIHARD2-dev:
      PLDA: (24.9390, 33.3898)
      SC:   (26.9699, 24.4945)

    AMI:
      PLDA: (9.11885, 9.05615)
      SC:   (6.39346, 7.28769)



'

currDir=$PWD
kaldiDir=/home/manoj/kaldi
expDir=$currDir/exp_kaldi
wavDir=$currDir/demo_wav
rttmDir=$currDir/demo_rttm
wavList=$currDir/wavList
readlink -f $wavDir/* > $wavList

# Extraction parameters
window=1.5
window_period=0.75
min_segment=0.5
nnetDir=$kaldiDir/egs/voxceleb/v2/exp/xvector_nnet_1a/
transformDir=$nnetDir/xvectors_train/

# Evaluation parameters
method=plda # plda or SC (spectral clustering)
useOracleNumSpkr=1
useCollar=1
skipDataPrep=1
dataDir=$expDir/data
nj=16

if [[ "$method" == "SC" ]] && [[ ! -d Auto-Tuning-Spectral-Clustering ]]; then
  echo "Please install https://github.com/tango4j/Auto-Tuning-Spectral-Clustering"
  exit 1
fi

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
  echo "Creating VAD files.."
  python convert_rttm_to_vad.py $wavDir $rttmDir $expDir/oracleVAD
  while read -r line; do
      uttID=`echo $line | cut -f 1 -d ' '`
      inVadFile=$expDir/oracleVAD/$uttID.csv # this change yet to be verified
      [ ! -f $inVadFile ] && { echo "Input vad file does not exist"; exit 0; }
      paste -d ' ' <(echo $uttID) <(cut -f 2 -d ',' $inVadFile | tr "\n" " " | sed "s/^/ [ /g" | sed "s/$/ ]/g") >> $dataDir/vad.txt
  done < $dataDir/utt2spk
  copy-vector ark,t:$dataDir/vad.txt ark,scp:$dataDir/vad.ark,$dataDir/vad.scp
  echo "Done"

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

  # Use extract.py or kaldi's extract_xvectors.sh
  diarization/nnet3/xvector/extract_xvectors.sh --nj $nj \
               --cmd "$train_cmd --mem 5G" \
               --window $window \
               --period $window_period \
               --apply-cmn false \
               --min-segment $min_segment \
               $nnetDir \
               $dataDir/segmented_cmn \
               $dataDir/kaldi_xvectors/

else
  [ ! -f $dataDir/kaldi_xvectors/xvector.scp ] && echo "Cannot find features" && exit 1;
fi

if [ "$method" == "plda" ]; then

  diarization/nnet3/xvector/score_plda.sh --nj $nj \
               --cmd "$train_cmd" \
               $transformDir \
               $dataDir/kaldi_xvectors \
               $expDir/plda/scoring

  diarization/cluster.sh --nj $nj \
               --cmd "$train_cmd --mem 5G" \
               --reco2num-spk $dataDir/reco2num_spk \
               $expDir/plda/scoring \
               $expDir/plda/clustering_oracleNumSpkr

  diarization/cluster.sh --nj $nj \
              --cmd "$train_cmd --mem 5G" \
              --threshold 0 \
              $expDir/plda/scoring \
              $expDir/plda/clustering_estNumSpkr

else

  # Compute the cosine affinity
  cd Auto-Tuning-Spectral-Clustering/sc_utils
  bash score_embedding.sh --cmd "$train_cmd" --nj 16 \
               --python_env ~/virtualenv/keras_fixed/bin/activate \
               --score_metric cos --out_dir $expDir/SC/cos_scores  \
               $dataDir/kaldi_xvectors $expDir/SC/cos_scores
  cd ..

  # Perform spectral clustering
  python spectral_opt.py --affinity_score_file $expDir/SC/cos_scores/scores.scp \
               --threshold 'None' --score_metric "cos" --max_speaker 10 \
               --spt_est_thres 'NMESC' --reco2num_spk $dataDir/reco2num_spk \
               --segment_file_input_path $dataDir/kaldi_xvectors/segments \
               --spk_labels_out_path $expDir/SC/labels_oracleNumSpkr \
               --sparse_search True
  mkdir -p $expDir/SC/clustering_oracleNumSpkr
  python sc_utils/make_rttm.py $dataDir/kaldi_xvectors/segments \
    $expDir/SC/labels_oracleNumSpkr $expDir/SC/clustering_oracleNumSpkr/rttm

  python spectral_opt.py --affinity_score_file $expDir/SC/cos_scores/scores.scp \
               --threshold 'None' --score_metric "cos" --max_speaker 10 \
               --spt_est_thres 'NMESC' \
               --segment_file_input_path $dataDir/kaldi_xvectors/segments \
               --spk_labels_out_path $expDir/SC/labels_estNumSpkr \
               --sparse_search True
  mkdir -p $expDir/SC/clustering_estNumSpkr
  python sc_utils/make_rttm.py $dataDir/kaldi_xvectors/segments \
    $expDir/SC/labels_estNumSpkr $expDir/SC/clustering_estNumSpkr/rttm
  cd ..


fi

# Evaluation
printf "DER with Oracle #Spkrs: "
rm -rf $currDir/oracle_ders.txt $currDir/oracle_rttms/
mkdir $currDir/oracle_rttms
cut -f 1 -d ' ' $dataDir/segments |\
while read -s wavid; do
    grep " ${wavid}-rec " $expDir/$method/clustering_oracleNumSpkr/rttm |\
    sed "s/-rec//g" > $currDir/oracle_rttms/$wavid.rttm
    der=`perl /home/manoj/kaldi/tools/sctk-2.4.10/bin/md-eval.pl $collarCmd \
    -r $rttmDir/$wavid.rttm -s $currDir/oracle_rttms/$wavid.rttm 2>&1 |\
    grep "OVERALL" | cut -f 2 -d '=' | cut -f 2 -d ' '`
  echo $der >> $currDir/oracle_ders.txt
done
meanDER=`awk '{sum += $1} END {print sum/NR}' $currDir/oracle_ders.txt`
echo $meanDER
rm -rf $currDir/oracle_ders.txt $cuttDir/oracle_rttms

printf "DER with Estimated #Spkrs: "
rm -rf $currDir/est_ders.txt $currDir/est_rttms/
mkdir $currDir/est_rttms
cut -f 1 -d ' ' $dataDir/segments |\
while read -s wavid; do
    grep " ${wavid}-rec " $expDir/$method/clustering_estNumSpkr/rttm |\
    sed "s/-rec//g" > $currDir/est_rttms/$wavid.rttm
    der=`perl /home/manoj/kaldi/tools/sctk-2.4.10/bin/md-eval.pl $collarCmd \
    -r $rttmDir/$wavid.rttm -s $currDir/est_rttms/$wavid.rttm 2>&1 |\
    grep "OVERALL" | cut -f 2 -d '=' | cut -f 2 -d ' '`
  echo $der >> $currDir/est_ders.txt
done
meanDER=`awk '{sum += $1} END {print sum/NR}' $currDir/est_ders.txt`
echo $meanDER
rm -rf $currDir/est_ders.txt $cuttDir/est_rttms

rm $wavList
