# Interpretable Differencing Rules of Magnetoencephalography Decoding Models

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
pip install einops
```

### Dataset Preprocessing

The CamCAN dataset can be downloaded from the Cambridge Centre for Ageing Neuroscience website at https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/.
The DecMeg2014 dataset is available at https://www.kaggle.com/c/decoding-the-human-brain/data.
The pre-processed training and test set are provided in https://drive.google.com/drive/folders/1d1xHb9bYQzoCcvlZ7mrFJvm570-MISvu.

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

### Performance of the Pre-trained Models

| Dataset    | RF    | MLP   | VARCNN | HGRN  | ATCNet |
| ---------- | ----- | ----- | ------ | ----- | ------ |
| CamCAN     | 90.29 | 94.36 | 95.66  | 95.17 | 96.16  |
| DecMeg2014 | 64.48 | 75.76 | 79.29  | 80.47 | 83.00  |

## Experimental Running

### Benchmark

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../
bash run.sh
```

### Hyperparameters Selection

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../
bash run_search.sh
```

### Ablation

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../
bash run_ablation.sh
```

### TABULAR DATASETS

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../
python tabular_difference_analysis.py
```

### Individual Model Pair Evaluation

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../

# for our proposed approach
python difference_analysis.py --cfg ../configs/CamCAN/Logit.yaml
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml

# you can also change settings at command line
python difference_analysis.py --cfg ../configs/CamCAN/Logit.yaml  MODEL_A mlp
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml  EXPLAINER.MAX_DEPTH 3
```

## Acknowledgement

