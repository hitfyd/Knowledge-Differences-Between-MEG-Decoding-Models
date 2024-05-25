#!/bin/bash

dataset='DecMeg2014'  # CamCAN or DecMeg2014
models=('mlp' 'sdt' 'lfcnn' 'varcnn' 'hgrn' 'atcnet') # 'mlp' 'sdt' 'lfcnn' 'varcnn' 'hgrn' 'atcnet'
explainer='Logit_Aug_Sel' # 'SS' 'IMD' 'Delta' 'Logit' 'Logit_Aug' 'Logit_Sel' 'Logit_Aug_Sel'

N=${#models[@]}

for (( a=0 ; a<N-1 ; $(( a++ )) ))
do
  for (( b=a+1; b<N; $(( b++)) ))
  do
    echo ${dataset} ${explainer} "${models[a]}" "${models[b]}"
    python difference_analysis.py --cfg ../configs/${dataset}/${explainer}.yaml MODEL_A "${models[a]}" MODEL_B "${models[b]}"
  done
done
