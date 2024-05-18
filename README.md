# Knowledge-Distillation-to-MEG-Glassbox-Models

## Experimental Preparation

### Experimental Environment

In the conde environment, do the following:

```
conda create -n <environment name> python=3.11
conda activate <environment name>
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm yacs
conda install numpy pandas scikit-learn matplotlib networkx graphviz pygraphviz
# The graphviz library needs to be installed on your operating system as well, see https://pygraphviz.github.io/documentation/stable/install.html
pip install -U "ray"
pip install Boruta
conda install -c conda-forge imbalanced-learn
pip install einops
conda install catboost
pip install merlinxai pyeda rulefit
```

Major dependency packages:

```
python==3.11.8
pytorch==2.2.1
tqdm==4.65.0
yacs==0.1.6
numpy==1.24.3
pandas==2.2.1
scikit-learn==1.3.0
matplotlib==3.8.0
networkx==3.1
graphviz==2.50.0
pygraphviz==1.9
ray==2.9.3
```

### Dataset Preprocessing

The CamCAN dataset can be downloaded from the Cambridge Centre for Ageing Neuroscience website at \url{https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/}.
The DecMeg2014 dataset is available at \url{https://www.kaggle.com/c/decoding-the-human-brain/data}.
The pre-processed training and test set are provided in \url{https://drive.google.com/drive/folders/1d1xHb9bYQzoCcvlZ7mrFJvm570-MISvu}.

CamCAN Preprocessing Script:
```angular2html
cd ./Dataset-Preprocessing-Script
python CamCAN2npz.py
python CamCAN2Dataset.py
```

DecMeg2014 Preprocessing Script:
```angular2html
cd ./Dataset-Preprocessing-Script
python DecMeg2Dataset.py
```

Generating the topographic map location information of the gradient sensors:
```angular2html
cd ./Dataset-Preprocessing-Script
python CreateGradChannelsInfo.py
```

### Baseline Performance of the Pre-trained Teacher Models

| Model\Dataset | CamCAN |                   | DecMeg2014 |                   |
| ----------- | ------ | ----------------- | ---------- | ----------------- |
|             | Loss   | Top-1 Accuracy(%) | Loss       | Top-1 Accuracy(%) |
| LFCNN       | 0.1167 | 95.6131           | 0.5895     | 81.6498           |
| VARCNN      | 0.1214 | 95.6640           | 0.6250     | 79.2929           |
| HGRN        | 0.1286 | 95.1897           | 0.5574     | 80.4714           |

## Experimental Running

### Evaluation on the CamCAN dataset

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../

# for instance, our FAKD approach.
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/varcnn_sdt.yaml

# you can also change settings at command line
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 128 ShapleyFAKD.M 2
```

### Evaluation on the DecMeg2014 dataset

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../

# for instance, our FAKD approach.
python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml

# you can also change settings at command line
python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 ShapleyFAKD.LOSS.FA_WEIGHT 100
```

### Validation of Feature Attribution Map Knowledge Transfer

```angular2html
cd ./Run-Script
python attribution.py
```

## Acknowledgement

1. https://github.com/megvii-research/mdistiller
2. https://github.com/xuyxu/Soft-Decision-Tree
