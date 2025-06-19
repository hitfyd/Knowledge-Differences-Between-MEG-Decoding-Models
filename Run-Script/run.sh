#!/bin/bash

dataset='CamCAN'  # CamCAN or DecMeg2014
models=('rf' 'mlp' 'lfcnn' 'varcnn' 'hgrn' 'atcnet' 'ctnet') # 'gnb' 'rf' 'lr' 'mlp' 'lfcnn' 'varcnn' 'hgrn' 'atcnet'
explainers=('Logit' 'Delta' 'MERLIN' 'SS' 'IMD')  # 'SS' 'IMD' 'Delta' 'MERLIN' 'Logit'

N_M=${#models[@]}
N_E=${#explainers[@]}

for (( a=0 ; a<N_M-1 ; $(( a++ )) ))
do
  for (( b=a+1 ; b<N_M ; $(( b++)) ))
  do
    for (( c=0 ; c<N_E ; $(( c++ )) ))
    do
      echo ${dataset} "${explainers[c]}" "${models[a]}" "${models[b]}"
      python meg_difference_analysis.py --cfg ../configs/${dataset}/"${explainers[c]}".yaml MODEL_A "${models[a]}" MODEL_B "${models[b]}"
    done
  done
done
