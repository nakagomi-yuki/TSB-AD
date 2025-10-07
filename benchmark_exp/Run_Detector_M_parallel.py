# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License
# Modified for OpenMP parallelization

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import multiprocessing as mp
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict

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

def process_single_file(args_tuple):
    """単一ファイルの処理関数（並列化用）"""
    (filename, dataset_dir, AD_Name, Optimal_Det_HP, score_dir, save_dir, save) = args_tuple
    
    target_dir = os.path.join(score_dir, AD_Name)
    
    # 既に処理済みのファイルをスキップ
    if os.path.exists(target_dir+'/'+filename.split('.')[0]+'.npy'):
        return None
    
    print(f'Processing: {filename} by {AD_Name}')
    
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
            output = run_Semisupervise_AD(AD_Name, data_train, data, **Optimal_Det_HP)
        elif AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(AD_Name, data, **Optimal_Det_HP)
        else:
            raise Exception(f"{AD_Name} is not defined")
        
        end_time = time.time()
        run_time = end_time - start_time
        
        if isinstance(output, np.ndarray):
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', output)
            result = {
                'filename': filename,
                'success': True,
                'run_time': run_time,
                'label_length': len(label),
                'error': None
            }
        else:
            result = {
                'filename': filename,
                'success': False,
                'run_time': run_time,
                'label_length': len(label),
                'error': output
            }
        
        # 評価結果の保存
        evaluation_result = None
        if save and isinstance(output, np.ndarray):
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                result['evaluation'] = evaluation_result
            except Exception as e:
                result['evaluation'] = None
                result['evaluation_error'] = str(e)
        
        return result
        
    except Exception as e:
        return {
            'filename': filename,
            'success': False,
            'run_time': 0,
            'label_length': 0,
            'error': str(e)
        }

if __name__ == '__main__':
    Start_T = time.time()
    
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score (Parallel Version)')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-M/')
    parser.add_argument('--file_lsit', type=str, default='../Datasets/File_List/TSB-AD-M-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/multi/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/multi/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--AD_Name', type=str, default='IForest')
    parser.add_argument('--n_jobs', type=int, default=cpu_count(), help='Number of parallel jobs')
    parser.add_argument('--parallel_type', type=str, default='process', choices=['process', 'thread'], 
                       help='Type of parallelization: process or thread')
    args = parser.parse_args()
    
    # ディレクトリの作成
    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ログ設定
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}_parallel.log', 
                       level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    file_list = pd.read_csv(args.file_lsit)['file_name'].values
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)
    print(f'Processing {len(file_list)} files with {args.n_jobs} parallel jobs')
    
    # 並列処理のための引数タプルを作成
    args_tuples = [(filename, args.dataset_dir, args.AD_Name, Optimal_Det_HP, 
                   args.score_dir, args.save_dir, args.save) for filename in file_list]
    
    write_csv = []
    
    # 並列処理の実行
    if args.parallel_type == 'process':
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            results = list(executor.map(process_single_file, args_tuples))
    else:
        with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            results = list(executor.map(process_single_file, args_tuples))
    
    # 結果の処理とログ出力
    successful_files = 0
    failed_files = 0
    
    for result in results:
        if result is None:  # スキップされたファイル
            continue
            
        if result['success']:
            successful_files += 1
            logging.info(f'Success at {result["filename"]} using {args.AD_Name} | '
                        f'Time cost: {result["run_time"]:.3f}s at length {result["label_length"]}')
            
            # 評価結果の保存
            if args.save and 'evaluation' in result and result['evaluation'] is not None:
                evaluation_result = result['evaluation']
                print(f'evaluation_result for {result["filename"]}: {evaluation_result}')
                list_w = list(evaluation_result.values())
                list_w.insert(0, result['run_time'])
                list_w.insert(0, result['filename'])
                write_csv.append(list_w)
                
                # CSV保存
                col_w = list(evaluation_result.keys())
                col_w.insert(0, 'Time')
                col_w.insert(0, 'file')
                w_csv = pd.DataFrame(write_csv, columns=col_w)
                w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)
        else:
            failed_files += 1
            logging.error(f'At {result["filename"]}: {result["error"]}')
    
    End_T = time.time()
    total_time = End_T - Start_T
    
    print(f'\n=== Processing Summary ===')
    print(f'Total files: {len(file_list)}')
    print(f'Successful: {successful_files}')
    print(f'Failed: {failed_files}')
    print(f'Skipped: {len(file_list) - successful_files - failed_files}')
    print(f'Total time: {total_time:.3f}s')
    print(f'Average time per file: {total_time/len(file_list):.3f}s')
    print(f'Parallelization type: {args.parallel_type}')
    print(f'Number of jobs: {args.n_jobs}')
    
    logging.info(f'Processing completed. Total time: {total_time:.3f}s, '
                f'Successful: {successful_files}, Failed: {failed_files}')
