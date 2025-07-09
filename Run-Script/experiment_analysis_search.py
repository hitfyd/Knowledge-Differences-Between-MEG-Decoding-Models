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
models = ['hgrn', 'atcnet'] # 'rf', 'mlp', 'lfcnn', 'varcnn', 'hgrn', 'atcnet', 'ctnet'
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
            # for matrix in evaluation_matrix.keys():
                # for max_depth in max_depth_list:
                #     for augment_factor in augment_factor_list:
                #         for selection_threshold in selection_threshold_list:
                #             key_id = f"{explainer}_{max_depth}_{augment_factor}_{selection_threshold}"
                #             result = evaluation_matrix[matrix][key_id]

            # 设置绘图风格
            sns.set_palette("tab10", n_colors=len(augment_factor_list))
            # 设置全局字体大小
            plt.rcParams['font.size'] = 13

            # 创建图表
            fig, axes = plt.subplots(
                len(evaluation_matrix), len(max_depth_list),
                figsize=(5*len(max_depth_list), 5*len(evaluation_matrix)), sharey='row',
                gridspec_kw={'hspace': 0.15, 'wspace': 0.1}  # 增加行间距，减少列间距
            )
            fig.suptitle(f'{explainer_name}: {model_names[models[i]]} vs {model_names[models[j]]} ({dataset})', fontsize=18, fontweight='bold', y=0.98)
            fig.subplots_adjust(top=0.9, bottom=0.15)
            # 处理单行/单列的特殊情况
            if len(evaluation_matrix) == 1:
                axes = axes[np.newaxis, :]
            if len(max_depth_list) == 1:
                axes = axes[:, np.newaxis]

            for matrix_id, matrix in enumerate(evaluation_matrix.keys()):
                # 遍历每个max_depth创建子图
                for idx, max_depth in enumerate(max_depth_list):
                    ax = axes[matrix_id, idx] if len(max_depth_list) > 1 else axes

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
                                       marker='*', markersize=6, linewidth=3,
                                       label=f'$g$={augment_factor}')

                        # 添加误差带（透明）
                        color = line[0].get_color()
                        ax.fill_between(selection_threshold_list,
                                        np.array(y_values) - np.array(y_errors),
                                        np.array(y_values) + np.array(y_errors),
                                        alpha=0.1, color=color)

                    # # 设置子图标题和标签
                    # ax.set_title('Max Depth $d_{max}$ = ' + str(max_depth), fontsize=14, pad=10)
                    # ax.set_xlabel('Selection Threshold $t$', fontsize=12)
                    # if idx == 0:
                    #     ax.set_ylabel(evaluation_matrix_name[matrix], fontsize=12)
                    # # 添加图例（只在第一个子图添加）
                    # if idx == 0:
                    #     ax.legend(title='Counterfactual Generation Factor', fontsize=10,
                    #               title_fontsize=11, frameon=True, shadow=True)

                    # 设置子图标题和标签
                    if matrix_id == 0:
                        ax.set_title(f'Max Depth $d_{{max}}$ = {max_depth}', fontsize=14, pad=12)
                    if idx == 0:
                        ax.set_ylabel(evaluation_matrix_name[matrix], fontsize=16, fontweight='bold', labelpad=12)
                    if matrix_id == len(evaluation_matrix) - 1:
                        ax.set_xlabel('Selection Threshold $t$', fontsize=14, labelpad=12)
                    # 在底部中央添加共享图例
                    if matrix_id == 0 and idx == 0:
                        fig.legend(
                            # handles=legend_handles,
                            # labels=legend_labels,
                            title='Counterfactual Generation Factor $g$',
                            fontsize=13,
                            title_fontsize=14,
                            frameon=True,
                            shadow=True,
                            loc='lower center',
                            bbox_to_anchor=(0.5, 0.01),  # 位于底部中央
                            ncol=min(6, len(augment_factor_list)),  # 自适应列数
                            columnspacing=1.0  # 列间距
                        )

            # 保存和显示
            save_dict = f'./output/{dataset}/'
            output_path =  f'{tags}_{models[i]}_{models[j]}_{explainer_name}'
            save_figure(fig, save_dict, output_path)
            plt.close(fig)  # 关闭图形释放内存
