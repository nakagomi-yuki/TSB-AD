# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License
# Modified for OpenMP parallelization

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import multiprocessing as mp
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Uni_algo_HP_dict

# OpenMP環境変数の設定
os.environ['OMP_NUM_THREADS'] = str(cpu_count())
os.environ['MKL_NUM_THREADS'] = str(cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count())

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())
print(f"Available CPU cores: {cpu_count()}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

def process_single_task(args_tuple):
    """単一タスク（ファイル+パラメータ組み合わせ）の処理関数（並列化用）"""
    (filename, dataset_dir, AD_Name, params, save_dir) = args_tuple
    
    try:
        file_path = os.path.join(dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        
        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]
        
        start_time = time.time()
        
        if AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(AD_Name, data_train, data, **params)
        elif AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(AD_Name, data, **params)
        else:
            raise Exception(f"{AD_Name} is not defined")
        
        end_time = time.time()
        run_time = end_time - start_time
        
        # 評価結果の取得
        try:
            evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
            result = {
                'filename': filename,
                'params': params,
                'success': True,
                'run_time': run_time,
                'evaluation': evaluation_result,
                'error': None
            }
        except Exception as e:
            result = {
                'filename': filename,
                'params': params,
                'success': False,
                'run_time': run_time,
                'evaluation': None,
                'error': f'Evaluation error: {str(e)}'
            }
        
        return result
        
    except Exception as e:
        return {
            'filename': filename,
            'params': params,
            'success': False,
            'run_time': 0,
            'evaluation': None,
            'error': str(e)
        }

if __name__ == '__main__':
    Start_T = time.time()
    
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='HP Tuning (Parallel Version)')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, default='../Datasets/File_List/TSB-AD-U-Tuning.csv')
    parser.add_argument('--save_dir', type=str, default='eval/HP_tuning/uni/')
    parser.add_argument('--AD_Name', type=str, default='IForest')
    parser.add_argument('--n_jobs', type=int, default=cpu_count(), help='Number of parallel jobs')
    parser.add_argument('--parallel_type', type=str, default='process', choices=['process', 'thread'], 
                       help='Type of parallelization: process or thread')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing tasks')
    args = parser.parse_args()
    
    # ディレクトリの作成
    os.makedirs(args.save_dir, exist_ok=True)
    
    file_list = pd.read_csv(args.file_lsit)['file_name'].values
    Det_HP = Uni_algo_HP_dict[args.AD_Name]
    
    keys, values = zip(*Det_HP.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_files = len(file_list)
    total_combinations = len(combinations)
    total_tasks = total_files * total_combinations
    
    print(f'Total files: {total_files}')
    print(f'Total parameter combinations: {total_combinations}')
    print(f'Total tasks: {total_tasks}')
    print(f'Processing with {args.n_jobs} parallel jobs')
    print('=' * 50)
    
    # 全タスクのリストを作成
    all_tasks = []
    for filename in file_list:
        for params in combinations:
            all_tasks.append((filename, args.dataset_dir, args.AD_Name, params, args.save_dir))
    
    write_csv = []
    successful_tasks = 0
    failed_tasks = 0
    
    # バッチ処理で並列実行
    batch_size = args.batch_size
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_tasks))
        batch_tasks = all_tasks[start_idx:end_idx]
        
        print(f'Processing batch {batch_idx + 1}/{total_batches} '
              f'(tasks {start_idx + 1}-{end_idx}/{len(all_tasks)})')
        
        # 並列処理の実行
        if args.parallel_type == 'process':
            with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                batch_results = list(executor.map(process_single_task, batch_tasks))
        else:
            with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
                batch_results = list(executor.map(process_single_task, batch_tasks))
        
        # 結果の処理
        for result in batch_results:
            if result['success']:
                successful_tasks += 1
                print(f'  Success: {result["filename"]} with params {result["params"]}')
                
                # CSVに追加
                evaluation_result = result['evaluation']
                list_w = list(evaluation_result.values())
                list_w.insert(0, result['params'])
                list_w.insert(0, result['filename'])
                write_csv.append(list_w)
            else:
                failed_tasks += 1
                print(f'  Failed: {result["filename"]} with params {result["params"]} - {result["error"]}')
        
        # バッチごとにCSVを保存
        if write_csv:
            col_w = list(evaluation_result.keys())
            col_w.insert(0, 'HP')
            col_w.insert(0, 'file')
            w_csv = pd.DataFrame(write_csv, columns=col_w)
            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)
        
        print(f'Batch {batch_idx + 1} completed. Progress: {end_idx}/{len(all_tasks)} '
              f'({(end_idx/len(all_tasks)*100):.1f}%)')
        print('-' * 50)
    
    End_T = time.time()
    total_time = End_T - Start_T
    
    print(f'\n=== HP Tuning Summary ===')
    print(f'Total tasks: {total_tasks}')
    print(f'Successful: {successful_tasks}')
    print(f'Failed: {failed_tasks}')
    print(f'Total time: {total_time:.3f}s')
    print(f'Average time per task: {total_time/total_tasks:.3f}s')
    print(f'Parallelization type: {args.parallel_type}')
    print(f'Number of jobs: {args.n_jobs}')
    print(f'Batch size: {args.batch_size}')
    
    # 最終結果の保存
    if write_csv:
        col_w = list(evaluation_result.keys())
        col_w.insert(0, 'HP')
        col_w.insert(0, 'file')
        w_csv = pd.DataFrame(write_csv, columns=col_w)
        w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}_final.csv', index=False)
        print(f'Results saved to: {args.save_dir}/{args.AD_Name}_final.csv')
