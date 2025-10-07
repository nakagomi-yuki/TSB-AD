# -*- coding: utf-8 -*-
# Author: Assistant
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RunDetectorAnalyzer:
    def __init__(self, algorithm_name, base_dir='eval/score/uni'):
        """
        実行結果解析クラス
        
        Parameters:
        -----------
        algorithm_name : str
            解析するアルゴリズム名
        base_dir : str
            結果ファイルのベースディレクトリ
        """
        self.algorithm_name = algorithm_name
        self.base_dir = base_dir
        self.score_dir = f'{base_dir}/{algorithm_name}'
        self.log_file = f'{self.score_dir}/000_run_{algorithm_name}.log'
        
        # 結果データの読み込み
        self.load_results()
    
    def load_results(self):
        """結果ファイルの読み込み"""
        self.score_files = []
        self.score_data = {}
        
        if os.path.exists(self.score_dir):
            # .npyファイルの検索
            npy_files = glob.glob(f'{self.score_dir}/*.npy')
            self.score_files = [os.path.basename(f) for f in npy_files]
            
            print(f"見つかったスコアファイル数: {len(self.score_files)}")
            
            # サンプルファイルの読み込み（最初の5ファイル）
            for i, filename in enumerate(self.score_files[:5]):
                try:
                    filepath = f'{self.score_dir}/{filename}'
                    scores = np.load(filepath)
                    self.score_data[filename] = scores
                    print(f"読み込み成功: {filename} (形状: {scores.shape})")
                except Exception as e:
                    print(f"読み込みエラー {filename}: {e}")
        else:
            print(f"ディレクトリが見つかりません: {self.score_dir}")
    
    def analyze_log_file(self):
        """ログファイルの解析"""
        if not os.path.exists(self.log_file):
            print(f"ログファイルが見つかりません: {self.log_file}")
            return None
        
        print(f"\n=== {self.algorithm_name} ログ解析 ===")
        
        with open(self.log_file, 'r') as f:
            log_lines = f.readlines()
        
        # 成功・失敗の統計
        success_count = 0
        error_count = 0
        total_time = 0
        
        for line in log_lines:
            if 'Success' in line:
                success_count += 1
                # 実行時間の抽出
                if 'Time cost:' in line:
                    try:
                        time_str = line.split('Time cost:')[1].split('s')[0].strip()
                        total_time += float(time_str)
                    except:
                        pass
            elif 'ERROR' in line or 'error' in line:
                error_count += 1
        
        print(f"成功: {success_count}")
        print(f"失敗: {error_count}")
        print(f"総実行時間: {total_time:.2f}秒")
        if success_count > 0:
            print(f"平均実行時間: {total_time/success_count:.2f}秒")
        
        return {
            'success_count': success_count,
            'error_count': error_count,
            'total_time': total_time,
            'avg_time': total_time/success_count if success_count > 0 else 0
        }
    
    def analyze_score_distribution(self):
        """異常スコアの分布解析"""
        if not self.score_data:
            print("スコアデータがありません")
            return
        
        print(f"\n=== {self.algorithm_name} スコア分布解析 ===")
        
        all_scores = []
        for filename, scores in self.score_data.items():
            all_scores.extend(scores.flatten())
        
        all_scores = np.array(all_scores)
        
        # 基本統計量
        stats = {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'median': np.median(all_scores),
            'q25': np.percentile(all_scores, 25),
            'q75': np.percentile(all_scores, 75)
        }
        
        print("基本統計量:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
        
        # 分布の可視化
        plt.figure(figsize=(12, 4))
        
        # ヒストグラム
        plt.subplot(1, 3, 1)
        plt.hist(all_scores, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{self.algorithm_name} - スコア分布')
        plt.xlabel('異常スコア')
        plt.ylabel('頻度')
        
        # ボックスプロット
        plt.subplot(1, 3, 2)
        plt.boxplot(all_scores)
        plt.title(f'{self.algorithm_name} - ボックスプロット')
        plt.ylabel('異常スコア')
        
        # Q-Qプロット
        plt.subplot(1, 3, 3)
        from scipy import stats as scipy_stats
        scipy_stats.probplot(all_scores, dist="norm", plot=plt)
        plt.title(f'{self.algorithm_name} - Q-Qプロット')
        
        plt.tight_layout()
        plt.savefig(f'{self.algorithm_name}_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats
    
    def compare_with_ground_truth(self, sample_file=None):
        """正解ラベルとの比較（サンプルファイルがある場合）"""
        if sample_file is None:
            print("正解ラベルとの比較にはサンプルファイルが必要です")
            return
        
        # データセットから正解ラベルを読み込み
        try:
            df = pd.read_csv(f'../Datasets/TSB-AD-U/{sample_file}')
            labels = df['Label'].values
            
            # 対応するスコアファイルを探す
            score_file = sample_file.replace('.csv', '.npy')
            if score_file in self.score_data:
                scores = self.score_data[score_file]
                
                print(f"\n=== {self.algorithm_name} 正解ラベル比較 ===")
                print(f"ファイル: {sample_file}")
                print(f"データ長: {len(labels)}")
                print(f"異常ラベル数: {np.sum(labels)}")
                print(f"異常率: {np.mean(labels)*100:.2f}%")
                
                # スコアとラベルの関係
                plt.figure(figsize=(12, 4))
                
                # 時系列プロット
                plt.subplot(1, 3, 1)
                plt.plot(scores, label='異常スコア', alpha=0.7)
                plt.plot(labels * np.max(scores), label='正解ラベル', alpha=0.7)
                plt.title(f'{self.algorithm_name} - 時系列比較')
                plt.xlabel('時間')
                plt.ylabel('スコア/ラベル')
                plt.legend()
                
                # スコア分布（正常vs異常）
                plt.subplot(1, 3, 2)
                normal_scores = scores[labels == 0]
                anomaly_scores = scores[labels == 1]
                plt.hist(normal_scores, bins=30, alpha=0.7, label='正常', density=True)
                plt.hist(anomaly_scores, bins=30, alpha=0.7, label='異常', density=True)
                plt.title(f'{self.algorithm_name} - 正常vs異常スコア分布')
                plt.xlabel('異常スコア')
                plt.ylabel('密度')
                plt.legend()
                
                # ROC曲線
                plt.subplot(1, 3, 3)
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='ランダム')
                plt.title(f'{self.algorithm_name} - ROC曲線')
                plt.xlabel('偽陽性率')
                plt.ylabel('真陽性率')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'{self.algorithm_name}_ground_truth_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                return {
                    'auc': roc_auc,
                    'normal_mean': np.mean(normal_scores),
                    'anomaly_mean': np.mean(anomaly_scores),
                    'separation': np.mean(anomaly_scores) - np.mean(normal_scores)
                }
        
        except Exception as e:
            print(f"正解ラベルとの比較でエラー: {e}")
            return None
    
    def generate_report(self, output_dir='analysis_reports'):
        """包括的レポートの生成"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== {self.algorithm_name} 包括的レポート生成 ===")
        
        # ログ解析
        log_stats = self.analyze_log_file()
        
        # スコア分布解析
        score_stats = self.analyze_score_distribution()
        
        # レポートの保存
        report = {
            'algorithm_name': self.algorithm_name,
            'log_analysis': log_stats,
            'score_analysis': score_stats,
            'files_processed': len(self.score_files)
        }
        
        with open(f'{output_dir}/{self.algorithm_name}_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # テキストレポート
        with open(f'{output_dir}/{self.algorithm_name}_report.txt', 'w') as f:
            f.write(f"TSB-AD 実行結果レポート\n")
            f.write(f"アルゴリズム: {self.algorithm_name}\n")
            f.write(f"生成日時: {pd.Timestamp.now()}\n")
            f.write(f"\n")
            
            if log_stats:
                f.write(f"実行統計:\n")
                f.write(f"  成功: {log_stats['success_count']}\n")
                f.write(f"  失敗: {log_stats['error_count']}\n")
                f.write(f"  総実行時間: {log_stats['total_time']:.2f}秒\n")
                f.write(f"  平均実行時間: {log_stats['avg_time']:.2f}秒\n")
                f.write(f"\n")
            
            if score_stats:
                f.write(f"スコア統計:\n")
                for key, value in score_stats.items():
                    f.write(f"  {key}: {value:.6f}\n")
                f.write(f"\n")
            
            f.write(f"処理ファイル数: {len(self.score_files)}\n")
        
        print(f"レポートが {output_dir} に保存されました")
        return report

def analyze_multiple_algorithms(algorithm_names, base_dir='eval/score/uni'):
    """複数アルゴリズムの比較解析"""
    print("=== 複数アルゴリズム比較解析 ===")
    
    results = {}
    
    for alg_name in algorithm_names:
        print(f"\n解析中: {alg_name}")
        try:
            analyzer = RunDetectorAnalyzer(alg_name, base_dir)
            results[alg_name] = analyzer.generate_report()
        except Exception as e:
            print(f"エラー {alg_name}: {e}")
    
    # 比較表の生成
    if results:
        comparison_data = []
        for alg_name, result in results.items():
            if result and 'log_analysis' in result:
                log_stats = result['log_analysis']
                comparison_data.append({
                    'Algorithm': alg_name,
                    'Success': log_stats['success_count'],
                    'Errors': log_stats['error_count'],
                    'Total_Time': log_stats['total_time'],
                    'Avg_Time': log_stats['avg_time'],
                    'Files_Processed': result['files_processed']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('algorithm_comparison.csv', index=False)
        
        print("\n=== アルゴリズム比較表 ===")
        print(comparison_df.to_string(index=False))
        
        # 可視化
        plt.figure(figsize=(15, 5))
        
        # 成功率
        plt.subplot(1, 3, 1)
        success_rates = comparison_df['Success'] / (comparison_df['Success'] + comparison_df['Errors'])
        plt.bar(comparison_df['Algorithm'], success_rates)
        plt.title('成功率比較')
        plt.ylabel('成功率')
        plt.xticks(rotation=45)
        
        # 平均実行時間
        plt.subplot(1, 3, 2)
        plt.bar(comparison_df['Algorithm'], comparison_df['Avg_Time'])
        plt.title('平均実行時間比較')
        plt.ylabel('時間（秒）')
        plt.xticks(rotation=45)
        
        # 処理ファイル数
        plt.subplot(1, 3, 3)
        plt.bar(comparison_df['Algorithm'], comparison_df['Files_Processed'])
        plt.title('処理ファイル数比較')
        plt.ylabel('ファイル数')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return results

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run_Detector_U.py 結果解析')
    parser.add_argument('--algorithm', type=str, default='IForest',
                       help='解析するアルゴリズム名')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='比較するアルゴリズムのリスト')
    parser.add_argument('--sample_file', type=str,
                       help='正解ラベル比較用のサンプルファイル')
    parser.add_argument('--output_dir', type=str, default='analysis_reports',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    if args.compare:
        # 複数アルゴリズムの比較
        analyze_multiple_algorithms(args.compare)
    else:
        # 単一アルゴリズムの解析
        analyzer = RunDetectorAnalyzer(args.algorithm)
        
        # ログ解析
        analyzer.analyze_log_file()
        
        # スコア分布解析
        analyzer.analyze_score_distribution()
        
        # 正解ラベルとの比較（サンプルファイルがある場合）
        if args.sample_file:
            analyzer.compare_with_ground_truth(args.sample_file)
        
        # レポート生成
        analyzer.generate_report(args.output_dir)

if __name__ == '__main__':
    main() 