# CamCAN
python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE SeparateSurrogate

python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE IMDExplainer

python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE DeltaExplainer

python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_hgrn.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/CamCAN/Before_Distill/varcnn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/KD_varcnn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/CamCAN/After_Distill/FAKD_varcnn_varcnn.yaml EXPLAINER.TYPE LogitDeltaRule


# DecMeg2014
python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE SeparateSurrogate
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE SeparateSurrogate

python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE IMDExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE IMDExplainer

python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE DeltaExplainer
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE DeltaExplainer

python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/lfcnn_varcnn.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/DecMeg2014/Before_Distill/hgrn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/KD_hgrn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_sdt.yaml EXPLAINER.TYPE LogitDeltaRule
python difference_analysis.py --cfg ../configs/DecMeg2014/After_Distill/FAKD_hgrn_hgrn.yaml EXPLAINER.TYPE LogitDeltaRule
