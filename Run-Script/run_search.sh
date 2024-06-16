#!/bin/bash

dataset='DecMeg2014'  # CamCAN or DecMeg2014
explainers=('Logit')  # 'SS' 'IMD' 'Delta' 'MERLIN' 'Logit'
depths=(4 5 6 7)
augment_factor=(0.0 0.5 1.0 2.0 4.0)
threshold=(0.0 2.0 4.0 6.0 8.0)

N_E=${#explainers[@]}
N_D=${#depths[@]}
N_F=${#augment_factor[@]}
N_T=${#threshold[@]}

# max rf-atcnet
model_a='rf'
model_b='atcnet'

for (( e=0 ; e<N_E ; $(( e++ )) ))
do
  for (( d=0 ; d<N_D ; $(( d++ )) ))
  do
    for (( f=0 ; f<N_F ; $(( f++ )) ))
    do
      for (( t=0 ; t<N_T ; $(( t++ )) ))
      do
        echo ${dataset} "${explainers[e]}" "${model_a}" "${model_b}" "${depths[d]}" "${augment_factor[f]}" "${threshold[t]}"
        python difference_analysis.py --cfg ../configs/${dataset}/"${explainers[e]}".yaml MODEL_A "${model_a}" MODEL_B "${model_b}" EXPLAINER.MAX_DEPTH "${depths[d]}" AUGMENT_FACTOR "${augment_factor[f]}" SELECTION.Diff.THRESHOLD "${threshold[t]}"
      done
    done
  done
done

## min varcnn-hgrn
#model_a='varcnn'
#model_b='hgrn'
