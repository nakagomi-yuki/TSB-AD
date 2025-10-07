# TSB-AD OpenMP並列化ガイド

このガイドでは、TSB-ADプロジェクトのRun_Detector.pyとHP_tuning.pyスクリプトをOpenMP対応で並列化する方法について説明します。

## 作成されたファイル

### 並列化スクリプト
- `Run_Detector_U_parallel.py` - 単変量異常検知の並列化バージョン
- `Run_Detector_M_parallel.py` - 多変量異常検知の並列化バージョン
- `HP_Tuning_U_parallel.py` - 単変量ハイパーパラメータチューニングの並列化バージョン
- `HP_Tuning_M_parallel.py` - 多変量ハイパーパラメータチューニングの並列化バージョン

### ヘルパーツール
- `openmp_helper.py` - OpenMP環境設定とチェック用ツール

## 主な改善点

### 1. ファイルレベルの並列化
- 複数のファイルを同時に処理
- ProcessPoolExecutorまたはThreadPoolExecutorを使用
- 既に処理済みのファイルを自動スキップ

### 2. ハイパーパラメータ組み合わせの並列化
- ファイル×パラメータ組み合わせの全タスクを並列処理
- バッチ処理でメモリ使用量を制御
- 進捗状況の詳細表示

### 3. OpenMP環境変数の最適化
- `OMP_NUM_THREADS`、`MKL_NUM_THREADS`等の自動設定
- CPUコア数に基づく最適なスレッド数の計算
- ハイパースレッディングを考慮した設定

### 4. エラーハンドリングとログ
- 詳細なエラーログと成功/失敗の統計
- 処理時間の測定とレポート
- バッチごとの進捗表示

## 使用方法

### 1. OpenMP環境の確認

```bash
# OpenMPサポートの確認
python openmp_helper.py --check

# 最適なスレッド数の確認
python openmp_helper.py
```

### 2. 並列化Detectorの実行

```bash
# 単変量異常検知（プロセス並列化）
python Run_Detector_U_parallel.py --AD_Name IForest --n_jobs 8 --parallel_type process

# 多変量異常検知（スレッド並列化）
python Run_Detector_M_parallel.py --AD_Name IForest --n_jobs 4 --parallel_type thread --save True
```

### 3. 並列化HP Tuningの実行

```bash
# 単変量ハイパーパラメータチューニング
python HP_Tuning_U_parallel.py --AD_Name IForest --n_jobs 8 --batch_size 50

# 多変量ハイパーパラメータチューニング
python HP_Tuning_M_parallel.py --AD_Name IForest --n_jobs 4 --batch_size 100
```

### 4. 起動スクリプトの作成

```bash
# 自動起動スクリプトの作成
python openmp_helper.py --create-launcher Run_Detector_U_parallel.py --set-threads 8

# 作成されたスクリプトの実行
./launch_Run_Detector_U_parallel.sh --AD_Name IForest
```

## パラメータ説明

### 共通パラメータ
- `--n_jobs`: 並列ジョブ数（デフォルト: CPUコア数）
- `--parallel_type`: 並列化タイプ（`process`または`thread`）

### Detector専用パラメータ
- `--dataset_dir`: データセットディレクトリ
- `--file_lsit`: ファイルリストCSV
- `--score_dir`: スコア保存ディレクトリ
- `--save_dir`: 評価結果保存ディレクトリ
- `--save`: 評価結果の保存フラグ
- `--AD_Name`: 異常検知アルゴリズム名

### HP Tuning専用パラメータ
- `--batch_size`: バッチサイズ（メモリ使用量制御）

## パフォーマンス最適化のヒント

### 1. スレッド数の調整
- **CPU集約的タスク**: CPUコア数と同数
- **I/O集約的タスク**: CPUコア数の2-4倍
- **メモリ制限**: 利用可能メモリに応じて調整

### 2. 並列化タイプの選択
- **Process並列化**: CPU集約的、メモリ使用量が多い場合
- **Thread並列化**: I/O集約的、メモリ使用量が少ない場合

### 3. バッチサイズの調整
- **メモリが多い場合**: 大きなバッチサイズ（100-500）
- **メモリが少ない場合**: 小さなバッチサイズ（10-50）

## トラブルシューティング

### 1. メモリ不足エラー
```bash
# バッチサイズを小さくする
python HP_Tuning_U_parallel.py --batch_size 10

# 並列ジョブ数を減らす
python Run_Detector_U_parallel.py --n_jobs 2
```

### 2. OpenMPライブラリの問題
```bash
# OpenMP環境の確認
python openmp_helper.py --check

# 環境変数の手動設定
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 3. デッドロックやハング
```bash
# スレッド並列化に切り替え
python script.py --parallel_type thread

# 並列ジョブ数を減らす
python script.py --n_jobs 2
```

## パフォーマンス比較

### 期待される高速化
- **ファイル並列化**: 2-8倍の高速化（CPUコア数に依存）
- **HP Tuning並列化**: 10-50倍の高速化（タスク数に依存）
- **OpenMP最適化**: 1.2-2倍の高速化（アルゴリズムに依存）

### ベンチマーク例
```bash
# 元のスクリプトとの比較
time python Run_Detector_U.py --AD_Name IForest
time python Run_Detector_U_parallel.py --AD_Name IForest --n_jobs 8
```

## 注意事項

1. **メモリ使用量**: 並列化によりメモリ使用量が増加する可能性があります
2. **ファイルI/O**: SSDの使用を推奨します
3. **ログファイル**: 並列処理によりログが混在する可能性があります
4. **再現性**: 並列処理により結果の順序が変わる可能性があります

## サポート

問題が発生した場合は、以下を確認してください：
1. OpenMPライブラリのインストール状況
2. システムのメモリとCPUリソース
3. ログファイルのエラーメッセージ
4. 環境変数の設定状況
