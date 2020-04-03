## <div align="center">Deep speaker embeddings in PyTorch</div>

### Requirements:
Python Libraries
```
python==3.6.10
torch==1.4.0
kaldi-io==0.9.1
kaldi-python-io==1.0.4
```


### Installation:
* Install the python libraries listed in [Requirements](#requirements)
* Install the Kaldi toolkit: https://github.com/kaldi-asr/kaldi/blob/master/INSTALL
* Download this repository. NOTE: Destination need not be inside Kaldi installation.
* Set the `voxcelebDir` variable inside [pytorch_run.sh](pytorch_run.sh)

### Data preparation

Training features are expected in Kaldi nnet3 egs format, and read using the `nnet3EgsDL` class defined in [train_utils.py](train_utils.py). The voxceleb recipe is provided in [pytorch_run.sh](pytorch_run.sh) to prepare them. Features for embedding extraction are expected in Kaldi matrix format, read using the [kaldi_io](https://github.com/vesis84/kaldi-io-for-python) library. Extracted embeddings are written in Kaldi vector format, similar to `xvector.ark`.

### Training
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_xent.py <egsDir>
```
```
usage: train_xent.py [-h] [--local_rank LOCAL_RANK] [-modelType MODELTYPE]
                     [-featDim FEATDIM] [-resumeTraining RESUMETRAINING]
                     [-resumeModelDir RESUMEMODELDIR]
                     [-numArchives NUMARCHIVES] [-numSpkrs NUMSPKRS]
                     [-logStepSize LOGSTEPSIZE] [-batchSize BATCHSIZE]
                     [-numEgsPerArk NUMEGSPERARK]
                     [-preFetchRatio PREFETCHRATIO]
                     [-optimMomentum OPTIMMOMENTUM] [-baseLR BASELR]
                     [-maxLR MAXLR] [-numEpochs NUMEPOCHS]
                     [-noiseEps NOISEEPS] [-pDropMax PDROPMAX]
                     [-stepFrac STEPFRAC]
                     egsDir

positional arguments:
  egsDir                Directory with training archives

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
  -modelType MODELTYPE  Refer train_utils.py
  -featDim FEATDIM      Frame-level feature dimension
  -resumeTraining RESUMETRAINING
                        (1) Resume training, or (0) Train from scratch
  -resumeModelDir RESUMEMODELDIR
                        Path containing training checkpoints
  -numArchives NUMARCHIVES
                        Number of egs.*.ark files
  -numSpkrs NUMSPKRS    Number of output labels
  -logStepSize LOGSTEPSIZE
                        Iterations per log
  -batchSize BATCHSIZE  Batch size
  -numEgsPerArk NUMEGSPERARK
                        Number of training examples per egs file
  -preFetchRatio PREFETCHRATIO
                        xbatchSize to fetch from dataloader
  -optimMomentum OPTIMMOMENTUM
                        Optimizer momentum
  -baseLR BASELR        Initial LR
  -maxLR MAXLR          Maximum LR
  -numEpochs NUMEPOCHS  Number of training epochs
  -noiseEps NOISEEPS    Noise strength before pooling
  -pDropMax PDROPMAX    Maximum dropout probability
  -stepFrac STEPFRAC    Training iteration when dropout = pDropMax

```
`egsDir` contains the nnet3 egs files.

### Embedding extraction
```
usage: extract.py [-h] [-modelType MODELTYPE] [-numSpkrs NUMSPKRS]
                  modelDirectory featDir embeddingDir

positional arguments:
  modelDirectory        Directory containing the model checkpoints
  featDir               Directory containing features ready for extraction
  embeddingDir          Output directory

optional arguments:
  -h, --help            show this help message and exit
  -modelType MODELTYPE  Refer train_utils.py
  -numSpkrs NUMSPKRS    Number of output labels for model
```
The script [pytorch_run.sh](pytorch_run.sh) can be used to train embeddings on the voxceleb recipe on an end-to-end basis.

## Pretrained model

#### 1. Speaker Verification
To reproduce voxceleb EER results with the pretrained model, follow the below steps.
NOTE: The voxceleb features must be prepared using `prepare_feats_for_egs.sh` prior to evaluation.

1) Download the [model](https://drive.google.com/file/d/1gbAWDdWN_pkOim4rWVXUlfuYjfyJqUHZ/view?usp=sharing)
2) Extract `models/` and `xvectors/` into the installation directory
3) Set the following variables in [pytorch_run.sh](pytorch_run.sh):
    ```
    modelDir=models/xvec_preTrained
    trainFeatDir=data/train_combined_no_sil
    trainXvecDir=xvectors/xvec_preTrained/train
    testFeatDir=data/voxceleb1_test_no_sil
    testXvecDir=xvectors/xvec_preTrained/test
    ```
4) Extract embeddings and compute EER, minDCF. Set `stage=7` in [pytorch_run.sh](pytorch_run.sh) and execute:
   ```
   bash pytorch_run.sh
   ```
5) Alternatively, pretrained PLDA model is available inside `xvectors/train` directory. Set `stage=9` in [pytorch_run.sh](pytorch_run.sh) and execute:
   ```
   bash pytorch_run.sh
   ```
#### 2. Speaker Diarization
Place the audio files to diarize and their corresponding RTTM files in `demo_wav/` and `demo_rttm/` directories. Execute:
```
bash diarize.sh
```

## Results

#### 1. Speaker Verification

##### Voxceleb1 test

|         | Kaldi           | pytorch_xvectors  |
|:-------------|:-------------:|:-----:|
| EER      | 3.128% | 2.815% |
| minDCF(p=0.01)      | 0.3258      |   0.3110 |
| minDCF(p=0.001) | 0.5003      |    0.4102 |

##### VOICES dev

|         | Kaldi           | pytorch_xvectors  |
|:-------------|:-------------:|:-----:|
| EER      | 10.3% | 8.591% |
| minDCF(p=0.01)      | 0.7845      |   0.6961 |
| minDCF(p=0.001) | 0.9406      |    0.8934 |


#### 2. Speaker Diarization (DER)

|         | Kaldi           | pytorch_xvectors  |
|:-------------|:-------------:|:-----:|
| DIHARD2 dev (no collar, oracle #spk)      | 25.38% | 26.48% |
| DIHARD2 dev (no collar, est #spk)      | 32.96% | 32.81% |
| AMI (171 meetings, collar, oracle #spk) | 10.94% | 15.62% |
| AMI (171 meetings, collar, est #spk) | 13.20% | 14.88% |
