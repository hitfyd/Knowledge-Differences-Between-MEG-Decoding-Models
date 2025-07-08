#!/bin/bash

dataset='DecMeg2014'  # CamCAN or DecMeg2014

# max rf-atcnet
model_a='rf'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"

# best hgrn-atcnet
model_a='hgrn'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"

dataset='CamCAN'  # CamCAN or DecMeg2014

# max rf-atcnet
model_a='rf'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"

# best varcnn-atcnet
model_a='varcnn'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"

# Search_SS

dataset='DecMeg2014'  # CamCAN or DecMeg2014

# max rf-atcnet
model_a='rf'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search_SS.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"

# best hgrn-atcnet
model_a='hgrn'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search_SS.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"

dataset='CamCAN'  # CamCAN or DecMeg2014

# max rf-atcnet
model_a='rf'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search_SS.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"

# best varcnn-atcnet
model_a='varcnn'
model_b='atcnet'

echo ${dataset} "${model_a}" "${model_b}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Search_SS.yaml MODEL_A "${model_a}" MODEL_B "${model_b}"
