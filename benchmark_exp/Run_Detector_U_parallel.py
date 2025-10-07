# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License
# Modified for OpenMP parallelization

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import multiprocessing as mp
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

# OpenMP環境変数の設定（各プロセス1スレッド）
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
    
    target_dir = os.path.join(score_dir, AD_Name + 'q')
    
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
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, default='../Datasets/File_List/TSB-AD-U-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/uni/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--AD_Name', type=str, default=os.environ.get('AD_Name', 'IForest'))
    parser.add_argument('--n_jobs', type=int, default=int(os.environ.get('n_jobs', cpu_count())), help='Number of parallel jobs')
# ThreadPoolExecutorは使用しないため、プロセス並列化のみ
    args = parser.parse_args()
    
    print(f'CPU cores: {cpu_count()}')
    print(f'Max processes: {args.n_jobs}')
    print(f'Threads per process: 1')
    print(f'Total logical threads: {args.n_jobs}')
    print(f'OMP_NUM_THREADS set to: {os.environ.get("OMP_NUM_THREADS")}')
    
    # ディレクトリの作成
    target_dir = os.path.join(args.score_dir, args.AD_Name + 'q')
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ログ設定
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}q.log', 
                       level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    file_list = pd.read_csv(args.file_lsit)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)
    print(f'Processing {len(file_list)} files with {args.n_jobs} parallel jobs')
    
    # 並列処理のための引数タプルを作成
    args_tuples = [(filename, args.dataset_dir, args.AD_Name, Optimal_Det_HP, 
                   args.score_dir, args.save_dir, args.save) for filename in file_list]
    
    write_csv = []
    
    # 並列処理の実行（プロセスのみ）
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
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
            
            # 評価結果の一時保存（CSV保存は後で一括処理）
            if args.save and 'evaluation' in result and result['evaluation'] is not None:
                evaluation_result = result['evaluation']
                print(f'evaluation_result for {result["filename"]}: {evaluation_result}')
                list_w = list(evaluation_result.values())
                list_w.insert(0, result['run_time'])
                list_w.insert(0, result['filename'])
                write_csv.append(list_w)
        else:
            failed_files += 1
            logging.error(f'At {result["filename"]}: {result["error"]}')
    
    # 評価結果の一括CSV保存（for文の外で一度だけ実行）
    if args.save and write_csv:
        print(f'Saving evaluation results to CSV file...')
        # write_csvが空でないことが確認されているので、最初の要素からカラム名を取得
        sample_result = None
        for r in results:
            if r is not None and 'evaluation' in r and r['evaluation'] is not None:
                sample_result = r
                break
        
        if sample_result is not None:
            col_w = list(sample_result['evaluation'].keys())
            col_w.insert(0, 'Time')
            col_w.insert(0, 'pdf_u')
            w_csv = pd.DataFrame(write_csv, columns=col_w)
            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}q.csv', index=False)
            print(f'Successfully saved {len(write_csv)} evaluation results to CSV')
        else:
            print('No valid evaluation results to save')
    
    End_T = time.time()
    total_time = End_T - Start_T
    
    print(f'\n=== Processing Summary ===')
    print(f'Total files: {len(file_list)}')
    print(f'Successful: {successful_files}')
    print(f'Failed: {failed_files}')
    print(f'Skipped: {len(file_list) - successful_files - failed_files}')
    print(f'Total time: {total_time:.3f}s')
    print(f'Average time per file: {total_time/len(file_list):.3f}s')
    print(f'Parallelization type: process only')
    print(f'Number of processes: {args.n_jobs}')
    
    logging.info(f'Processing completed. Total time: {total_time:.3f}s, '
                f'Successful: {successful_files}, Failed: {failed_files}')
