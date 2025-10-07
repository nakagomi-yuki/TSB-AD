#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrequencyBasedADpl80.csvの結果評価スクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_analyze_results():
    """FrequencyBasedADの結果を読み込んで分析"""
    
    # CSVファイルの読み込み
    csv_path = Path("eval/metrics/uni/FrequencyBasedADpl80.csv")
    
    if not csv_path.exists():
        print(f"エラー: {csv_path}が見つかりません")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"データ読み込み完了: {len(df)}ファイル")
    print(f"カラム: {list(df.columns)}")
    print("\n" + "="*80)
    
    return df

def basic_statistics(df):
    """基本統計の計算"""
    print("📊 基本統計サマリー")
    print("="*50)
    
    # 数値カラムのみを抽出
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Time']  # Timeを除外
    
    # 各メトリクスの基本統計
    stats_df = df[numeric_cols].describe()
    
    print("\n🎯 主要評価メトリクスの統計:")
    key_metrics = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'Standard-F1']
    
    for metric in key_metrics:
        if metric in stats_df.columns:
            mean_val = stats_df.loc['mean', metric]
            std_val = stats_df.loc['std', metric]
            min_val = stats_df.loc['min', metric]
            max_val = stats_df.loc['max', metric]
            median_val = stats_df.loc['50%', metric]
            
            print(f"\n{metric}:")
            print(f"  平均: {mean_val:.4f} ± {std_val:.4f}")
            print(f"  中央値: {median_val:.4f}")
            print(f"  範囲: [{min_val:.4f}, {max_val:.4f}]")
    
    print("\n⏱️ 処理時間統計:")
    time_stats = df['Time'].describe()
    print(f"  平均処理時間: {time_stats['mean']:.3f}秒")
    print(f"  中央値処理時間: {time_stats['50%']:.3f}秒")
    print(f"  最小処理時間: {time_stats['min']:.3f}秒")
    print(f"  最大処理時間: {time_stats['max']:.3f}秒")

def performance_analysis(df):
    """パフォーマンス分析"""
    print("\n🚀 パフォーマンス分析")
    print("="*50)
    
    # AUC-PRの分布分析
    auc_pr_scores = df['AUC-PR']
    print(f"\n📈 AUC-PR分析:")
    print(f"  AUC-PR > 0.5: {(auc_pr_scores > 0.5).sum()}/{len(df)} files ({100*(auc_pr_scores > 0.5).mean():.1f}%)")
    print(f"  AUC-PR > 0.8: {(auc_pr_scores > 0.8).sum()}/{len(df)} files ({100*(auc_pr_scores > 0.8).mean():.1f}%)")
    print(f"  AUC-PR > 0.9: {(auc_pr_scores > 0.9).sum()}/{len(df)} files ({100*(auc_pr_scores > 0.9).mean():.1f}%)")
    
    # ROC-AUCの分析
    auc_roc_scores = df['AUC-ROC']
    print(f"\n📊 AUC-ROC分析:")
    print(f"  AUC-ROC = 0.5 (ランダム): {(auc_roc_scores == 0.5).sum()}/{len(df)} files ({100*(auc_roc_scores == 0.5).mean():.1f}%)")
    print(f"  AUC-ROC > 0.7: {(auc_roc_scores > 0.7).sum()}/{len(df)} files ({100*(auc_roc_scores > 0.7).mean():.1f}%)")
    
    # VUS-PRの分析
    vus_pr_scores = df['VUS-PR']
    print(f"\n🔍 VUS-PR分析:")
    print(f"  VUS-PR > 0.1: {(vus_pr_scores > 0.1).sum()}/{len(df)} files ({100*(vus_pr_scores > 0.1).mean():.1f}%)")
    
    # F1-Sコアの分析
    f1_scores = df['Standard-F1']
    print(f"\n⭐ Standard-F1分析:")
    print(f"  F1 > 0.5: {(f1_scores > 0.5).sum()}/{len(df)} files ({100*(f1_scores > 0.5).mean():.1f}%)")

def dataset_category_analysis(df):
    """データセットカテゴリ別分析"""
    print("\n📂 データセットカテゴリ別分析")
    print("="*50)
    
    # ファイル名からカテゴリを抽出
    df['category'] = df['pdf_u'].str.extract(r'(\w+)_id_')[0]
    categories = df['category'].unique()
    
    print(f"検出されたカテゴリ: {categories}")
    
    for category in categories:
        cat_data = df[df['category'] == category]
        print(f"\n📁 {category} ({len(cat_data)} files):")
        
        mean_auc_pr = cat_data['AUC-PR'].mean()
        mean_auc_roc = cat_data['AUC-ROC'].mean()
        mean_f1 = cat_data['Standard-F1'].mean()
        mean_time = cat_data['Time'].mean()
        
        print(f"  平均AUC-PR: {mean_auc_pr:.4f}")
        print(f"  平均AUC-ROC: {mean_auc_roc:.4f}")
        print(f"  平均F1: {mean_f1:.4f}")
        print(f"  平均処理時間: {mean_time:.3f}秒")

def extreme_cases_analysis(df):
    """極端なケースの分析"""
    print("\n🔍 極端なケースの分析")
    print("="*50)
    
    # 最高パフォーマンス
    print("🥇 最高パフォーマンス (AUC-PR順):")
    top_cases = df.nlargest(5, 'AUC-PR')
    for i, (_, row) in enumerate(top_cases.iterrows()):
        print(f"  {i+1}. {row['pdf_u']}: AUC-PR={row['AUC-PR']:.4f}, F1={row['Standard-F1']:.4f}")
    
    # 最低パフォーマンス
    print("\n🥉 最低パフォーマンス (AUC-PR順):")
    bottom_cases = df.nsmallest(5, 'AUC-PR')
    for i, (_, row) in enumerate(bottom_cases.iterrows()):
        print(f"  {i+1}. {row['pdf_u']}: AUC-PR={row['AUC-PR']:.4f}, F1={row['Standard-F1']:.4f}")
    
    # 長時間処理
    print("\n⏰ 長時間処理ケース (処理時間順):")
    slow_cases = df.nlargest(5, 'Time')
    for i, (_, row) in enumerate(slow_cases.iterrows()):
        print(f"  {i+1}. {row['pdf_u']}: {row['Time']:.1f}秒, AUC-PR={row['AUC-PR']:.4f}")

def overall_assessment(df):
    """総合評価"""
    print("\n🎯 FrequencyBasedAD 総合評価")
    print("="*60)
    
    # 全体統計
    total_files = len(df)
    mean_auc_pr = df['AUC-PR'].mean()
    mean_auc_roc = df['AUC-ROC'].mean()
    mean_f1 = df['Standard-F1'].mean()
    
    print(f"📈 総合パフォーマンス:")
    print(f"  処理ファイル数: {total_files}")
    print(f"  平均AUC-PR: {mean_auc_pr:.4f}")
    print(f"  平均AUC-ROC: {mean_auc_roc:.4f}")
    print(f"  平均F1-Score: {mean_f1:.4f}")
    
    # パフォーマンス評価
    good_performance = ((df['AUC-PR'] > 0.7) | (df['Standard-F1'] > 0.5)).sum()
    medium_performance = ((df['AUC-PR'] > 0.3) & (df['AUC-PR'] <= 0.7)).sum()
    poor_performance = ((df['AUC-PR'] <= 0.3) | (df['Standard-F1'] == 0)).sum()
    
    print(f"\n📊 パフォーマンス別ファイル数:")
    print(f"  良好 (AUC-PR>0.7 or F1>0.5): {good_performance} ({100*good_performance/total_files:.1f}%)")
    print(f"  中程度 (AUC-PR 0.3-0.7): {medium_performance} ({100*medium_performance/total_files:.1f}%)")
    print(f"  低調 (AUC-PR≤0.3 or F1=0): {poor_performance} ({100*poor_performance/total_files:.1f}%)")
    
    # 推奨事項
    print(f"\n💡 推奨事項:")
    if mean_auc_pr < 0.3:
        print("  - AUC-PRが低いため、ハイパーパラメータの再チューニングを推奨")
    if mean_f1 < 0.2:
        print("  - F1-Scoreが低いため、閾値設定やアノマリ検出ロジックの見直しを推奨")
    if (df['AUC-ROC'] == 0.5).sum() > total_files * 0.5:
        print("  - 多くのファイルで AUC-ROC = 0.5 の場合、アルゴリズムの根本的見直しが必要")
    
    # 効率性評価
    avg_time = df['Time'].mean()
    print(f"\n⚡ 効率性評価:")
    print(f"  平均処理時間: {avg_time:.3f}秒/ファイル")
    if avg_time > 10:
        print("  - 処理時間が長いため、計算効率の改善を検討")
    else:
        print("  - 処理時間は許容範囲内")

if __name__ == "__main__":
    print("🔬 FrequencyBasedAD 結果評価レポート")
    print("="*60)
    
    # データ読み込み
    df = load_and_analyze_results()
    
    if df is not None:
        # 各種分析を実行
        basic_statistics(df)
        performance_analysis(df)
        dataset_category_analysis(df)
        extreme_cases_analysis(df)
        overall_assessment(df)
        
        print(f"\n✅ 分析完了！合計 {len(df)} ファイルを評価しました。")
    else:
        print("❌ データの読み込みに失敗しました。")
