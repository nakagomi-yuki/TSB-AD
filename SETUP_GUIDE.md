# TSB-AD セットアップガイド

このガイドでは、TSB-ADプロジェクトを新しいサーバーにセットアップする手順を説明します。

## 前提条件

- Git
- Conda (Anaconda または Miniconda)
- Python 3.8以上 (推奨: 3.11)

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/TheDatumOrg/TSB-AD.git
cd TSB-AD
```

### 2. Conda環境の作成

```bash
conda create -n TSB-AD python=3.11
conda activate TSB-AD
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

**注意:** PyTorchのインストールで問題が発生した場合は、以下のコマンドを試してください:

```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 4. パッケージのインストール

```bash
pip install -e .
```

### 5. データセットのダウンロード

データセットは別途ダウンロードする必要があります:

- **TSB-AD-U**: https://www.thedatum.org/datasets/TSB-AD-U.zip
- **TSB-AD-M**: https://www.thedatum.org/datasets/TSB-AD-M.zip

ダウンロード後、以下のディレクトリに展開してください:

```bash
# ダウンロードしたzipファイルを展開
unzip TSB-AD-U.zip -d Datasets/
unzip TSB-AD-M.zip -d Datasets/
```

または、提供されているスクリプトを使用:

```bash
bash setup_datasets.sh
```

### 6. 動作確認

基本的な使用例を実行して、セットアップが正しく完了したか確認します:

```bash
python -m TSB_AD.main --AD_Name IForest
```

## ベンチマーク実験の実行

### ハイパーパラメータチューニング

単変量データ用:
```bash
cd benchmark_exp
python HP_Tuning_U.py
```

多変量データ用:
```bash
python HP_Tuning_M.py
```

### 並列実行版

```bash
python HP_Tuning_U_parallel.py
python HP_Tuning_M_parallel.py
```

### 検出器の実行

```bash
python Run_Detector_U.py
python Run_Detector_M.py
```

## ディレクトリ構造

```
TSB-AD/
├── TSB_AD/              # メインパッケージ
│   ├── models/          # 異常検知アルゴリズム
│   ├── evaluation/      # 評価メトリクス
│   └── utils/           # ユーティリティ関数
├── benchmark_exp/       # ベンチマーク実験スクリプト
├── Datasets/            # データセット (ダウンロード後)
│   ├── TSB-AD-U/        # 単変量データ
│   ├── TSB-AD-M/        # 多変量データ
│   └── File_List/       # データ分割リスト
├── requirements.txt     # 依存パッケージ
└── setup.py            # インストールスクリプト

```

## トラブルシューティング

### PyTorchのバージョン問題

現在のプロジェクトはPyTorch 2.3.0を使用しています。一部のFoundation Models (例: Chronos)はPyTorch 2.6以上を必要とする場合がありますが、互換性の問題により現在はサポートされていません。

### メモリ不足

大規模なデータセットを扱う場合、メモリ不足が発生する可能性があります。その場合は:
- バッチサイズを小さくする
- 並列実行のワーカー数を減らす
- スワップメモリを増やす

### CUDA関連のエラー

GPUを使用する場合、CUDAのバージョンとPyTorchのバージョンが一致していることを確認してください。

## 連絡先

問題や質問がある場合は、以下にお問い合わせください:
- Qinghua Liu (liu.11085@osu.edu)
- John Paparrizos (paparrizos.1@osu.edu)

または、GitHubのIssuesに投稿してください。
