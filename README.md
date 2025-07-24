# Interpretable Differencing Rules of Magnetoencephalography Decoding Models

## Experimental Preparation

### Experimental Environment

In the conde environment, do the following:

```
conda create -n <environment name> python=3.11
conda activate <environment name>
pip install torch torchvision torchaudio
conda install tqdm yacs
conda install numpy pandas scikit-learn matplotlib networkx graphviz pygraphviz
# The graphviz library needs to be installed on your operating system as well, see https://pygraphviz.github.io/documentation/stable/install.html
pip install -U "ray"
pip install einops
pip install braindecode==0.7 torchinfo
pip install pyeda rulefit pydot stopit apyori graphviz arff
pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.4.*" "dask-cudf-cu12==25.4.*" "cuml-cu12==25.4.*" "cugraph-cu12==25.4.*" "nx-cugraph-cu12==25.4.*" "cuspatial-cu12==25.4.*" "cuproj-cu12==25.4.*" "cuxfilter-cu12==25.4.*" "cucim-cu12==25.4.*" "pylibraft-cu12==25.4.*" "raft-dask-cu12==25.4.*" "cuvs-cu12==25.4.*" "nx-cugraph-cu12==25.4.*"
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

| Dataset    | RF    | VARCNN | HGRN  | ATCNet |
| ---------- | ----- | ------ | ----- | ------ |
| CamCAN     | 90.29 | 95.66  | 95.17 | 96.16  |
| DecMeg2014 | 64.48 | 79.29  | 80.47 | 83.00  |

## Experimental Running

### Benchmark

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../
bash run_benchmark.sh
```

### Hyperparameters Selection

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../
bash run_search.sh
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
python meg_difference_analysis.py --cfg ../configs/CamCAN/Logit.yaml
python meg_difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml

# you can also change settings at command line
python meg_difference_analysis.py --cfg ../configs/CamCAN/Logit.yaml  MODEL_A rf
python meg_difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml  EXPLAINER.MAX_DEPTH 3
```

## Acknowledgement

