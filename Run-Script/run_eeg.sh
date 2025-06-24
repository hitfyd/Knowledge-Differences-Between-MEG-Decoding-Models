#!/bin/bash

dataset='BCIIV2a'
models=('eegnetv4' 'eegnetv1')
explainers=('Logit' 'Delta' 'SS' 'IMD' 'MERLIN')  # 'Logit' 'Delta' 'SS' 'IMD' 'MERLIN'

N_M=${#models[@]}
N_E=${#explainers[@]}

for (( a=0 ; a<N_M-1 ; $(( a++ )) ))
do
  for (( b=a+1 ; b<N_M ; $(( b++)) ))
  do
    echo ${dataset} "${models[a]}" "${models[b]}"
    python meg_difference_analysis.py --cfg ../configs/${dataset}/Benchmark.yaml MODEL_A "${models[a]}" MODEL_B "${models[b]}"
  done
done
