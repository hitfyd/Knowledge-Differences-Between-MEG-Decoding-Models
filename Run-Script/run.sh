# CamCAN
#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE SeparateSurrogate

#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE IMDExplainer

#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE DeltaExplainer
#
#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE LogitDeltaRule


# DecMeg2014
#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE SeparateSurrogate

#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE IMDExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE IMDExplainer

#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE DeltaExplainer
#
#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
#python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE LogitDeltaRule

#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A mlp MODEL_B sdt
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A mlp MODEL_B lfcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A mlp MODEL_B varcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A mlp MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A mlp MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A sdt MODEL_B lfcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A sdt MODEL_B varcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A sdt MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A sdt MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A lfcnn MODEL_B varcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A lfcnn MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A lfcnn MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A varcnn MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A varcnn MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Delta.yaml MODEL_A hgrn MODEL_B atcnet

#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A mlp MODEL_B sdt
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A mlp MODEL_B lfcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A mlp MODEL_B varcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A mlp MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A mlp MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B lfcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B varcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B varcnn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B hgrn
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B atcnet
#python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A hgrn MODEL_B atcnet

python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B lfcnn AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B varcnn AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B hgrn AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B atcnet AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B varcnn AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B hgrn AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B atcnet AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B hgrn AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B atcnet AUGMENTATION BASE
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A hgrn MODEL_B atcnet AUGMENTATION BASE

python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B lfcnn SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B varcnn SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B hgrn SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B atcnet SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B varcnn SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B hgrn SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B atcnet SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B hgrn SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B atcnet SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A hgrn MODEL_B atcnet SELECTION.TYPE DiffShapley

python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B lfcnn AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B varcnn AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B hgrn AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A sdt MODEL_B atcnet AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B varcnn AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B hgrn AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A lfcnn MODEL_B atcnet AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B hgrn AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A varcnn MODEL_B atcnet AUGMENTATION BASE SELECTION.TYPE DiffShapley
python difference_analysis.py --cfg ../configs/DecMeg2014/Logit.yaml MODEL_A hgrn MODEL_B atcnet AUGMENTATION BASE SELECTION.TYPE DiffShapley
