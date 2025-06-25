#!/bin/bash

dataset='BCIIV2a'
models=('eegnetv4' 'eegnetv1')
augment=('Counterfactual' 'NONE')  # 'Counterfactual' 'NONE'

N_M=${#models[@]}
N_A=${#augment[@]}

for (( a=0 ; a<N_M-1 ; $(( a++ )) ))
do
  for (( b=a+1 ; b<N_M ; $(( b++)) ))
  do
    for (( c=0 ; c<N_A ; $(( c++ )) ))
    do
      echo ${dataset} "${models[a]}" "${models[b]}"
      python meg_difference_analysis.py --cfg ../configs/${dataset}/Benchmark.yaml MODEL_A "${models[a]}" MODEL_B "${models[b]}" AUGMENTATION "${augment[c]}"
    done
  done
done
