#!/bin/bash

dataset='DecMeg2014'  # CamCAN or DecMeg2014
explainers=('Logit_Aug' 'Logit_Aug_Sel')  # 'SS' 'IMD' 'Delta' 'MERLIN' 'Logit' 'Logit_Aug' 'Logit_Sel' 'Logit_Aug_Sel'
depths=(5 6 7)

N_D=${#depths[@]}
N_E=${#explainers[@]}

# max rf-atcnet
model_a='rf'
model_b='atcnet'

for (( a=0 ; a<N_D ; $(( a++ )) ))
do
  for (( c=0 ; c<N_E ; $(( c++ )) ))
  do
    echo ${dataset} "${explainers[c]}" "${model_a}" "${model_b}" "${depths[a]}"
    python difference_analysis.py --cfg ../configs/${dataset}/"${explainers[c]}".yaml MODEL_A "${model_a}" MODEL_B "${model_b}" EXPLAINER.MAX_DEPTH "${depths[a]}"
  done
done

# min varcnn-hgrn
model_a='varcnn'
model_b='hgrn'

for (( a=0 ; a<N_D ; $(( a++ )) ))
do
  for (( c=0 ; c<N_E ; $(( c++ )) ))
  do
    echo ${dataset} "${explainers[c]}" "${model_a}" "${model_b}" "${depths[a]}"
    python difference_analysis.py --cfg ../configs/${dataset}/"${explainers[c]}".yaml MODEL_A "${model_a}" MODEL_B "${model_b}" EXPLAINER.MAX_DEPTH "${depths[a]}"
  done
done
