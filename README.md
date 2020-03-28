## <div align="center">Deep speaker embeddings in PyTorch</div>

### Requirements:
Python Libraries
```
torch==1.4.0
python==3.6.10
kaldiio==2.15.1
kaldi-python-io==1.0.4
```
Kaldi: https://github.com/kaldi-asr/kaldi

### Installation:
* Install the Kaldi toolkit: https://github.com/kaldi-asr/kaldi/blob/master/INSTALL
* Download this repository. NOTE: Destination need not be inside Kaldi installation.
* Set the `voxcelebDir` variable inside [pytorch_run.sh](pytorch_run.sh)

### Data preparation

Training features are expected in Kaldi nnet3 egs format, and read using the `nnet3EgsDL` class defined in [train_utils.py](train_utils.py). The voxceleb recipe is provided in [pytorch_run.sh](pytorch_run.sh) to prepare them. Features for embedding extraction are expected in Kaldi matrix format, read using the [kaldi_io](https://github.com/vesis84/kaldi-io-for-python) library. Extracted embeddings are written in Kaldi vector format, similar to `xvector.ark`. 

### Training
``` 
python train_xent.py local.config
```

### Embedding extraction
```
python extract.py local.config
```
The script [pytorch_run.sh](pytorch_run.sh) can be used to train embeddings on the voxceleb recipe on an end-to-end basis.

## Configuration file 

Two models are defined in [models.py](models.py): 
* `simpleTDNN` (`modelType` = 3): A small time-delay neural network with stats pooling similar to the xvector architecture.
* `xvecTDNN` (`modelType` = 4): The xvector architecture as provided by Kaldi.

Three parameters have to be manually provided in the config file: number of training archives (`egs.*.ark`), number of speakers and number of examples in an archive.

## Pretrained model

To reproduce voxceleb EER results with the pretrained model, follow the below steps. 
NOTE: The voxceleb features must be prepared prior to evaluation.

1) Download the [model](https://drive.google.com/file/d/13kYPc76pA4_Axw0Jm5Vk0fLxcLD8oBWv/view?usp=sharing)
2) Extract local.config and models/ into the installation directory
3) Set the variables `trainFeatDir` and `testFeatDir` in local.config; and `voxceleb1_trials` in [pytorch_run.sh](pytorch_run.sh)
4) Extract embeddings and compute EER, minDCF. Set `stage=8` in [pytorch_run.sh](pytorch_run.sh) and execute:
   ```
   bash pytorch_run.sh 
   ```
5) Alternatively, pretrained PLDA model is available inside `xvectors/` directory.   
   
#### Results

|         | Kaldi           | pytorch_spkembed  |
|:-------------:|:-------------:|:-----:|
| EER      | 3.128% | 2.815% |
| minDCF(p=0.01)      | 0.3258      |   0.3110 |
| minDCF(p=0.001) | 0.5003      |    0.4102 |

