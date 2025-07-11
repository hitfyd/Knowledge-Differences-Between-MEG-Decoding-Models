from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr, spearmanr, kendalltau


@dataclass
class Result:
    mean: float
    std: float
    all: list[float]
    # mean_std: str
    # mean_std_latex: str
    # best_bold: bool
    # p_value: float


def str2num_list(s:str):
    num_list = [float(x) for x in s[1:-1].split()]
    return num_list


# dataset = 'BCIIV2a' # DecMeg2014    CamCAN  BCIIV2a
# models = ['eegnetv4', 'eegnetv1'] # 'eegnetv4', 'eegnetv1'

dataset = 'DecMeg2014' # DecMeg2014    CamCAN  BCIIV2a
tags = "Search"
models = ['rf', 'atcnet'] # 'rf', 'mlp', 'lfcnn', 'varcnn', 'hgrn', 'atcnet', 'ctnet'
model_names= {
    'rf': 'Random Forest',
    'varcnn': 'VARCNN',
    'hgrn': 'HGRN',
    'atcnet': 'ATCNet',
}
evaluation_matrix = {'test_f1': {},
                     'num_rules': {}
                     }
evaluation_matrix_name = {'test_f1': 'F1 Score',
                          'num_rules': 'Number of Rules',
                          }
explainer_dict = {'Logit': 'BO-RPPD',
                  # 'SS': 'Separate Surrogates',
                  }
max_depth_list = [5]
augment_factor_list = [0.0, 3.0]
selection_threshold_list = [0.0, 3.0]   # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
compared_id = [5, 0.0, 0.0]

# 计算皮尔逊相关性系数
x = [0.46, 0.51, 0.5, 0.56]
y = [17.4, 25.4, 19.2, 25.8]
corr, p_value = kendalltau(x, y)
print(corr, p_value)

for i in range(len(models)-1):
    for j in range(i+1, len(models)):
        record_file = f'./output/{dataset}/{tags}_{models[i]}_{models[j]}_record.csv'
        all_record_mean_std = pd.read_csv(record_file, encoding="utf_8_sig")

        # 读出所有实验记录
        for index, row in all_record_mean_std.iterrows():
            explainer = row['explainer']
            max_depth = row['max_depth']
            augment_factor = row['augment_factor']
            selection_threshold = row['selection_threshold']
            key_id = f"{explainer}_{max_depth}_{augment_factor}_{selection_threshold}"

            for matrix in evaluation_matrix.keys():
                print(dataset, key_id, matrix)
                matrix_list = f'{matrix}_list'
                assert matrix_list in all_record_mean_std.columns.tolist()
                mean, _, std = row[matrix].split(' ')
                results_list = str2num_list(row[matrix_list])
                evaluation_matrix[matrix][key_id] = Result(mean=float(mean), std=float(std), all=results_list)
                if [max_depth, augment_factor, selection_threshold] == compared_id:
                    evaluation_matrix[matrix][f"{explainer}_compared_id"] = Result(mean=float(mean), std=float(std), all=results_list)

        for explainer, explainer_name in explainer_dict.items():
            for matrix_id, matrix in enumerate(evaluation_matrix.keys()):
                compared_all = evaluation_matrix[matrix][f"{explainer}_compared_id"].all
                for max_depth in max_depth_list:
                    for augment_factor in augment_factor_list:
                        for selection_threshold in selection_threshold_list:
                            key_id = f"{explainer}_{max_depth}_{augment_factor}_{selection_threshold}"
                            result = evaluation_matrix[matrix][key_id]
                            p_value = ttest_ind(compared_all, result.all).pvalue
                            mean_std_latex = f"{result.mean} \pm {result.std}"
                            if p_value < 0.01:
                                mean_std_latex = '$' + mean_std_latex + '^{**}$'
                            elif p_value < 0.05:
                                mean_std_latex = '$' + mean_std_latex + '^{*}$'
                            else:
                                mean_std_latex = '$' + mean_std_latex + '$'
                            print(key_id, matrix, mean_std_latex)
                            evaluation_matrix[matrix][key_id + '_latex'] = mean_std_latex
