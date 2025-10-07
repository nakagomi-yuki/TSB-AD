# -*- coding: utf-8 -*-
# Author: Assistant
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import os
import glob
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

def generate_metrics_from_scores(algorithm_name, dataset_dir='../Datasets/TSB-AD-U/', 
                                score_dir='eval/score/uni/', save_dir='eval/metrics/uni/'):
    """
    既存のスコアファイルから評価指標を生成
    
    Parameters:
    -----------
    algorithm_name : str
        アルゴリズム名
    dataset_dir : str
        データセットディレクトリ
    score_dir : str
        スコアファイルのディレクトリ
    save_dir : str
        評価指標の保存ディレクトリ
    """
    
    # ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    
    # スコアファイルの検索
    score_path = f'{score_dir}/{algorithm_name}'
    if not os.path.exists(score_path):
        print(f"スコアディレクトリが見つかりません: {score_path}")
        return
    
    npy_files = glob.glob(f'{score_path}/*.npy')
    if not npy_files:
        print(f"スコアファイルが見つかりません: {score_path}")
        return
    
    print(f"処理するファイル数: {len(npy_files)}")
    
    results = []
    
    for npy_file in npy_files:
        try:
            # ファイル名から元のCSVファイル名を推測
            base_name = os.path.basename(npy_file).replace('.npy', '')
            
            # 対応するCSVファイルを探す
            csv_file = None
            for csv_path in glob.glob(f'{dataset_dir}/*.csv'):
                if base_name in os.path.basename(csv_path):
                    csv_file = csv_path
                    break
            
            if csv_file is None:
                print(f"CSVファイルが見つかりません: {base_name}")
                continue
            
            # データとラベルの読み込み
            df = pd.read_csv(csv_file).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            labels = df['Label'].astype(int).to_numpy()
            
            # スコアの読み込み
            scores = np.load(npy_file)
            
            # スライディングウィンドウの計算
            slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
            
            # 評価指標の計算
            evaluation_result = get_metrics(scores, labels, slidingWindow=slidingWindow)
            
            # 結果の保存
            result_row = [base_name] + list(evaluation_result.values())
            results.append(result_row)
            
            print(f"処理完了: {base_name}")
            
        except Exception as e:
            print(f"エラー {os.path.basename(npy_file)}: {e}")
            continue
    
    if results:
        # 結果をDataFrameに変換
        columns = ['file'] + list(evaluation_result.keys())
        results_df = pd.DataFrame(results, columns=columns)
        
        # CSVファイルに保存
        output_file = f'{save_dir}/{algorithm_name}.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"\n評価指標が保存されました: {output_file}")
        print(f"処理したファイル数: {len(results)}")
        
        # 統計情報の表示
        print("\n=== 評価指標の統計 ===")
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'file':
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
        
        return results_df
    else:
        print("処理できるファイルが見つかりませんでした")
        return None

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='既存スコアファイルから評価指標を生成')
    parser.add_argument('--algorithm', type=str, required=True,
                       help='アルゴリズム名')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-U/',
                       help='データセットディレクトリ')
    parser.add_argument('--score_dir', type=str, default='eval/score/uni/',
                       help='スコアファイルディレクトリ')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/',
                       help='評価指標保存ディレクトリ')
    
    args = parser.parse_args()
    
    print(f"アルゴリズム: {args.algorithm}")
    print(f"データセットディレクトリ: {args.dataset_dir}")
    print(f"スコアディレクトリ: {args.score_dir}")
    print(f"保存ディレクトリ: {args.save_dir}")
    
    results = generate_metrics_from_scores(
        args.algorithm,
        args.dataset_dir,
        args.score_dir,
        args.save_dir
    )
    
    if results is not None:
        print(f"\n評価指標の生成が完了しました！")

if __name__ == '__main__':
    main() 