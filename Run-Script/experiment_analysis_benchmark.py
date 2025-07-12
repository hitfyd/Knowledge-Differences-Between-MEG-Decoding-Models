from dataclasses import dataclass

import pandas as pd
from scipy.stats import ttest_ind


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

dataset = 'CamCAN' # DecMeg2014    CamCAN  BCIIV2a
tags = "Benchmark"
models = ['rf', 'atcnet']
models = ['rf', 'varcnn', 'hgrn', 'atcnet']
model_names= {
    'rf': 'Random Forest',
    'varcnn': 'VARCNN',
    'hgrn': 'HGRN',
    'atcnet': 'ATCNet',
}
evaluation_matrix = {'test_f1': {},
                     'num_rules': {}
                     }
evaluation_matrix_best = {'test_f1': 0.0,
                          'num_rules': float('inf'),
                          }
evaluation_matrix_name = {'test_f1': 'F1 Score',
                          'num_rules': 'Number of Rules',
                          }
explainer_dict = {
    'Delta': 'DeltaXpainer',
    'SS': 'Separate Surrogates',
    'IMD': 'IMD',
    'MERLIN': 'MERLIN',
    'Logit': 'BO-RPPD',
}
max_depth_list = [5]
augment_factor_list = [3.0]
selection_threshold_list = [3.0]
compared_id = ['Logit', 5, 3.0, 3.0]

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
            key_id = f"{explainer}_{models[i]}_{models[j]}_{max_depth}_{augment_factor}_{selection_threshold}"

            for matrix in evaluation_matrix.keys():
                print(dataset, key_id, matrix)
                matrix_list = f'{matrix}_list'
                assert matrix_list in all_record_mean_std.columns.tolist()
                mean, _, std = row[matrix].split(' ')
                results_list = str2num_list(row[matrix_list])
                evaluation_matrix[matrix][key_id] = Result(mean=float(mean), std=float(std), all=results_list)
                if [explainer, max_depth, augment_factor, selection_threshold] == compared_id:
                    evaluation_matrix[matrix][f"{models[i]}_{models[j]}_compared_id"] = Result(mean=float(mean), std=float(std), all=results_list)

for matrix_id, matrix in enumerate(evaluation_matrix.keys()):
    for max_depth in max_depth_list:
        for augment_factor in augment_factor_list:
            for selection_threshold in selection_threshold_list:
                for i in range(len(models) - 1):
                    for j in range(i + 1, len(models)):
                        best_mean, best_mean_id = evaluation_matrix_best[matrix], None
                        for explainer, explainer_name in explainer_dict.items():
                            compared_all = evaluation_matrix[matrix][f"{models[i]}_{models[j]}_compared_id"].all
                            key_id = f"{explainer}_{models[i]}_{models[j]}_{max_depth}_{augment_factor}_{selection_threshold}"
                            result = evaluation_matrix[matrix][key_id]
                            if evaluation_matrix_best[matrix] == 0.0:
                                if best_mean_id is None or result.mean > best_mean:
                                    best_mean = result.mean
                                    best_mean_id = key_id
                            if evaluation_matrix_best[matrix] == float('inf'):
                                if best_mean_id is None or result.mean < best_mean:
                                    best_mean = result.mean
                                    best_mean_id = key_id
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
                        # 更新最佳表现的latex格式加粗
                        best_mean_std_latex = evaluation_matrix[matrix][best_mean_id + '_latex']
                        best_mean_std_latex = '$\\bold{' + best_mean_std_latex[1:-1] + '}$'
                        evaluation_matrix[matrix][best_mean_id + '_latex'] = best_mean_std_latex

# 输出Markdown表格形式
for matrix_id, matrix in enumerate(evaluation_matrix.keys()):
    for explainer, explainer_name in explainer_dict.items():
        row_str = f'|{explainer_name}|'
        for i in range(len(models) - 1):
            for j in range(i + 1, len(models)):
                max_depth, augment_factor, selection_threshold = compared_id[1:]
                key_id = f"{explainer}_{models[i]}_{models[j]}_{max_depth}_{augment_factor}_{selection_threshold}_latex"
                row_str = row_str + f'{evaluation_matrix[matrix][key_id]}' + '|'
        print(row_str)

# 输出Latex表格形式
for matrix_id, matrix in enumerate(evaluation_matrix.keys()):
    for explainer, explainer_name in explainer_dict.items():
        row_str = f'{explainer_name}'
        for i in range(len(models) - 1):
            for j in range(i + 1, len(models)):
                max_depth, augment_factor, selection_threshold = compared_id[1:]
                key_id = f"{explainer}_{models[i]}_{models[j]}_{max_depth}_{augment_factor}_{selection_threshold}_latex"
                row_str = row_str + '&' + f'{evaluation_matrix[matrix][key_id]}'
        row_str = row_str + '\\\\'
        print(row_str)
