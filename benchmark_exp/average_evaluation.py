#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrequencyBasedADの評価結果の平均を計算するスクリプト
"""

import pandas as pd
import numpy as np
import os

def calculate_average_evaluation():
    """FrequencyBasedADの評価結果の平均を計算"""
    
    # 結果ファイルを読み込み
    results_file = 'eval/metrics/uni/FrequencyBasedAD.csv'
    
    try:
        if not os.path.exists(results_file):
            print(f"結果ファイル {results_file} が見つかりません")
            print("まず、Run_Detector_U.pyを実行してください")
            return None
            
        results_df = pd.read_csv(results_file)
        print(f"読み込んだデータセット数: {len(results_df)}")
        
        # 数値列のみを選択
        numeric_columns = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'Standard-F1', 'PA-F1', 'Event-based-F1', 'R-based-F1', 'Affiliation-F']
        
        # 平均を計算
        average_results = {}
        print("\n各指標の平均値:")
        print("=" * 50)
        
        for col in numeric_columns:
            if col in results_df.columns:
                # NaN値を除外して平均を計算
                valid_values = results_df[col].dropna()
                if len(valid_values) > 0:
                    average_results[col] = valid_values.mean()
                    std_val = valid_values.std()
                    print(f"{col:20s}: {average_results[col]:.4f} ± {std_val:.4f} (有効データ: {len(valid_values)}/{len(results_df)})")
                else:
                    average_results[col] = 0.0
                    print(f"{col:20s}: 0.0000 (有効データなし)")
            else:
                average_results[col] = 0.0
                print(f"{col:20s}: 0.0000 (列が存在しません)")
        
        # 実行時間の平均も計算
        if 'Time' in results_df.columns:
            average_results['Average_Time'] = results_df['Time'].mean()
            std_time = results_df['Time'].std()
            print(f"{'Average_Time':20s}: {average_results['Average_Time']:.3f} ± {std_time:.3f} seconds")
        
        average_results['Total_Datasets'] = len(results_df)
        
        print("\n" + "=" * 60)
        print("FrequencyBasedAD 平均評価結果 (50データセット)")
        print("=" * 60)
        for metric, value in average_results.items():
            if metric == 'Total_Datasets':
                print(f"{metric:20s}: {int(value)}")
            elif metric == 'Average_Time':
                print(f"{metric:20s}: {value:.3f} seconds")
            else:
                print(f"{metric:20s}: {value:.4f}")
        
        # 結果をCSVに保存
        avg_df = pd.DataFrame([average_results])
        avg_df.to_csv('FrequencyBasedAD_average_results.csv', index=False)
        print(f"\n結果を FrequencyBasedAD_average_results.csv に保存しました")
        
        # 性能の解釈
        print("\n" + "=" * 60)
        print("性能の解釈:")
        print("=" * 60)
        
        auc_roc = average_results.get('AUC-ROC', 0)
        auc_pr = average_results.get('AUC-PR', 0)
        f1 = average_results.get('Standard-F1', 0)
        
        if auc_roc < 0.6:
            print("⚠️  AUC-ROC < 0.6: ランダムレベル以下の性能")
        elif auc_roc < 0.7:
            print("⚠️  AUC-ROC < 0.7: 低い性能")
        elif auc_roc < 0.8:
            print("✅ AUC-ROC < 0.8: 中程度の性能")
        else:
            print("🎉 AUC-ROC >= 0.8: 良好な性能")
            
        if auc_pr < 0.1:
            print("⚠️  AUC-PR < 0.1: 非常に低い性能")
        elif auc_pr < 0.3:
            print("⚠️  AUC-PR < 0.3: 低い性能")
        elif auc_pr < 0.5:
            print("✅ AUC-PR < 0.5: 中程度の性能")
        else:
            print("🎉 AUC-PR >= 0.5: 良好な性能")
            
        if f1 < 0.1:
            print("⚠️  F1 < 0.1: 非常に低い性能")
        elif f1 < 0.3:
            print("⚠️  F1 < 0.3: 低い性能")
        elif f1 < 0.5:
            print("✅ F1 < 0.5: 中程度の性能")
        else:
            print("🎉 F1 >= 0.5: 良好な性能")
        
        return average_results
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

if __name__ == '__main__':
    calculate_average_evaluation()

