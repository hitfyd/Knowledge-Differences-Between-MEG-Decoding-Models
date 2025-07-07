#!/bin/bash

dataset='DecMeg2014'  # CamCAN or DecMeg2014

augment_factor=0.0
threshold=0.0
echo ${dataset} "${augment_factor}" "${threshold}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Ablation.yaml AUGMENT_FACTOR "${augment_factor[f]}" SELECTION.Diff.THRESHOLD "${threshold[t]}"

augment_factor=3.0
threshold=0.0
echo ${dataset} "${augment_factor}" "${threshold}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Ablation.yaml AUGMENT_FACTOR "${augment_factor[f]}" SELECTION.Diff.THRESHOLD "${threshold[t]}"

augment_factor=0.0
threshold=6.0
echo ${dataset} "${augment_factor}" "${threshold}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Ablation.yaml AUGMENT_FACTOR "${augment_factor[f]}" SELECTION.Diff.THRESHOLD "${threshold[t]}"

augment_factor=3.0
threshold=6.0
echo ${dataset} "${augment_factor}" "${threshold}"
python meg_difference_analysis.py --cfg ../configs/${dataset}/Ablation.yaml AUGMENT_FACTOR "${augment_factor[f]}" SELECTION.Diff.THRESHOLD "${threshold[t]}"
