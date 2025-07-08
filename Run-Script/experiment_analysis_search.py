import dataclasses
from dataclasses import dataclass

import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from differlib.engine.utils import save_figure


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
    'atcnet': 'ATCNet',
}
evaluation_matrix = {'test_f1': {},
                     'num_rules': {}
                     }
evaluation_matrix_name = {'test_f1': 'F1 Score',
                          'num_rules': 'Number of Rules',
                          }
explainer_dict = {'Logit': 'BO-RPPD',
                  'Delta': 'DeltaXpainer',
                  }
max_depth_list = [4, 5, 6, 7]
augment_factor_list = [0.0, 1.0, 3.0, 5.0]
selection_threshold_list = [0.0, 1.0, 3.0, 5.0]   # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

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

        for explainer, explainer_name in explainer_dict.items():
            for matrix in evaluation_matrix.keys():
                # for max_depth in max_depth_list:
                #     for augment_factor in augment_factor_list:
                #         for selection_threshold in selection_threshold_list:
                #             key_id = f"{explainer}_{max_depth}_{augment_factor}_{selection_threshold}"
                #             result = evaluation_matrix[matrix][key_id]

                # 设置绘图风格
                # plt.style.use('seaborn-v0_8-whitegrid')
                sns.set_palette("tab10")

                # 创建图表
                fig, axes = plt.subplots(1, len(max_depth_list), figsize=(20, 6), sharey=True)
                fig.suptitle(f'{explainer_name}: {model_names[models[i]]} vs {model_names[models[j]]} ({dataset})', fontsize=16, fontweight='bold')
                fig.subplots_adjust(top=0.85, bottom=0.15, wspace=0.1)

                # 遍历每个max_depth创建子图
                for idx, max_depth in enumerate(max_depth_list):
                    ax = axes[idx] if len(max_depth_list) > 1 else axes

                    # 为每个augment_factor绘制折线
                    for augment_factor in augment_factor_list:
                        # 收集当前参数组合的所有结果
                        y_values = []
                        y_errors = []

                        for selection_threshold in selection_threshold_list:
                            key_id = f"{explainer}_{max_depth}_{augment_factor}_{selection_threshold}"
                            result = evaluation_matrix[matrix][key_id]
                            y_values.append(result.mean)
                            y_errors.append(result.std)

                        # 绘制折线图和误差带
                        line = ax.plot(selection_threshold_list, y_values,
                                       marker='o', markersize=6, linewidth=2,
                                       label=f'$g$={augment_factor}')

                        # 添加误差带（透明）
                        color = line[0].get_color()
                        ax.fill_between(selection_threshold_list,
                                        np.array(y_values) - np.array(y_errors),
                                        np.array(y_values) + np.array(y_errors),
                                        alpha=0.2, color=color)

                    # 设置子图标题和标签
                    ax.set_title('Max Depth $d_{max}$ = ' + str(max_depth), fontsize=14, pad=10)
                    ax.set_xlabel('Selection Threshold $t$', fontsize=12)
                    if idx == 0:
                        ax.set_ylabel(evaluation_matrix_name[matrix], fontsize=12)

                    # # 设置网格和刻度
                    # ax.grid(True, linestyle='--', alpha=0.7)
                    # ax.set_xticks(selection_threshold_list)
                    # ax.xaxis.set_major_locator(MultipleLocator(1))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.05))

                    # 设置Y轴范围
                    # ax.set_ylim(0.3, 0.7)

                    # 添加图例（只在第一个子图添加）
                    if idx == 0:
                        ax.legend(title='Counterfactual Generation Factor', fontsize=10,
                                  title_fontsize=11, frameon=True, shadow=True)

                # 添加整体标题和标签
                plt.figtext(0.5, 0.02, f'Evaluation Metric: {evaluation_matrix_name[matrix]}', ha='center', fontsize=12)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # 保存和显示
                save_dict = f'./output/{dataset}/'
                output_path =  f'{tags}_{models[i]}_{models[j]}_{explainer_name}_{evaluation_matrix_name[matrix]}.svg'
                save_figure(fig, save_dict, output_path)
                plt.show()
