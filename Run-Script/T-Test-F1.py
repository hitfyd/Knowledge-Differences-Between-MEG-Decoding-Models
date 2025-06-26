import pandas as pd
from scipy.stats import ttest_ind

def str2num_list(s:str):
    num_list = [float(x) for x in s[1:-1].split()]
    return num_list

# dataset = 'BCIIV2a' # DecMeg2014    CamCAN  BCIIV2a
# models = ['eegnetv4', 'eegnetv1'] # 'eegnetv4', 'eegnetv1'

dataset = 'DecMeg2014' # DecMeg2014    CamCAN  BCIIV2a
models = ['rf', 'mlp', 'varcnn', 'hgrn', 'atcnet'] # 'rf', 'mlp', 'lfcnn', 'varcnn', 'hgrn', 'atcnet', 'ctnet'

record_id, compared_id = 0, 0
benchmarks = {record_id+1: 'DeltaXpainer',
              record_id+2: 'Separate Surrogates',
              record_id+3: 'IMD',
              record_id+4: 'MERLIN',
              record_id: 'RPPD',
              }
evaluation_matrix = {'test-f1': {},
                     'num-rules': {}
                     }

for i in range(len(models)-1):
    for j in range(i+1, len(models)):
        record_file = f'./output/{dataset}/{models[i]}_{models[j]}_record.csv'
        all_record_mean_std = pd.read_csv(record_file, encoding="utf_8_sig")

        for matrix in evaluation_matrix.keys():
            print(dataset, models[i], models[j], matrix)
            matrix_list = f'{matrix}-list'
            assert matrix_list in all_record_mean_std.columns.tolist()
            compared_results_list = str2num_list(all_record_mean_std[matrix_list][compared_id])
            for record_id, benchmark in benchmarks.items():
                mean, _, std = all_record_mean_std[matrix][record_id].split(' ')
                results_list = str2num_list(all_record_mean_std[matrix_list][record_id])
                p_value = ttest_ind(compared_results_list, results_list).pvalue
                if p_value < 0.01:
                    result = '$' + mean+' \pm '+ std + '^{**}$'
                elif p_value < 0.05:
                    result = '$' + mean+' \pm '+ std + '^{*}$'
                else:
                    result = '$' + mean+' \pm '+ std + '$'
                print(benchmark, result)#, results_list, f'{p_value:.4f}')
                evaluation_matrix[matrix][record_id] = result

        # 输出Markdown表格形式
        print('|', 'Methods', '|', '|'.join(evaluation_matrix.keys()), '|')
        print('|', '------', '|',  '|'.join(['----------' for _ in evaluation_matrix]), '|')
        for record_id, benchmark in benchmarks.items():
            print('|', benchmark, '|', '|'.join([evaluation_matrix[matrix][record_id] for matrix in evaluation_matrix.keys()]), '|')

        # 输出Latex表格形式
        for record_id, benchmark in benchmarks.items():
            print(benchmark, '&', '&'.join([evaluation_matrix[matrix][record_id] for matrix in evaluation_matrix.keys()]), '\\\\')
