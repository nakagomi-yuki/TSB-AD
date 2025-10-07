# クイックスタートガイド

TSB-ADを素早く始めるための最小限の手順です。

## 5分でセットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/TSB-AD.git
cd TSB-AD
```

### 2. 環境構築

```bash
conda create -n TSB-AD python=3.11 -y
conda activate TSB-AD
pip install -r requirements.txt
pip install -e .
```

### 3. データセットのダウンロード (オプション)

```bash
bash setup_datasets.sh
```

または手動で:

```bash
cd Datasets
wget https://www.thedatum.org/datasets/TSB-AD-U.zip
unzip TSB-AD-U.zip
```

### 4. 動作確認

```bash
python -m TSB_AD.main --AD_Name IForest
```

## 基本的な使い方

### Python スクリプトで使用

```python
import pandas as pd
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.evaluation.metrics import get_metrics

# データの読み込み
df = pd.read_csv('Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv').dropna()
data = df.iloc[:, 0:-1].values.astype(float)
label = df['Label'].astype(int).to_numpy()

# 異常検知の実行
output = run_Unsupervise_AD('IForest', data)

# 評価
results = get_metrics(output, label)
print(f"VUS-PR: {results['VUS-PR']:.4f}")
print(f"VUS-ROC: {results['VUS-ROC']:.4f}")
```

### 利用可能なアルゴリズム

```python
# 統計的手法
algorithms = [
    'IForest', 'OCSVM', 'LOF', 'KNN', 'PCA',
    'MatrixProfile', 'KMeansAD', 'HBOS', 'COPOD'
]

# ニューラルネットワーク
nn_algorithms = [
    'AutoEncoder', 'LSTMAD', 'CNN', 'USAD',
    'AnomalyTransformer', 'TranAD', 'TimesNet'
]

# Foundation Models
fm_algorithms = [
    'OFA', 'Chronos', 'MOMENT', 'TimesFM'
]
```

### ベンチマーク実験の実行

```bash
cd benchmark_exp

# ハイパーパラメータチューニング (単変量)
python HP_Tuning_U.py

# 検出器の実行 (単変量)
python Run_Detector_U.py

# 並列実行版
python HP_Tuning_U_parallel.py
python Run_Detector_U_parallel.py
```

## よく使うコマンド

### 環境の有効化

```bash
conda activate TSB-AD
```

### 特定のアルゴリズムでテスト

```bash
python -m TSB_AD.main --AD_Name IForest
python -m TSB_AD.main --AD_Name AutoEncoder
python -m TSB_AD.main --AD_Name LSTMAD
```

### 結果の確認

```bash
# 評価結果の確認
ls benchmark_exp/eval/score/uni/

# メトリクスの確認
ls benchmark_exp/eval/metrics/uni/
```

## 次のステップ

- 詳細なセットアップ: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- デプロイ方法: [DEPLOYMENT.md](DEPLOYMENT.md)
- ベンチマーク実験: [benchmark_exp/README.md](benchmark_exp/README.md)
- 公式ドキュメント: [README.md](README.md)

## トラブルシューティング

### PyTorchのインストールエラー

```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### メモリ不足

バッチサイズを小さくするか、より小さいモデルを使用してください。

### CUDA エラー

```bash
# CPU版のPyTorchを使用
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

## ヘルプ

問題が発生した場合:
1. [Issues](https://github.com/TheDatumOrg/TSB-AD/issues) を確認
2. 新しいIssueを作成
3. メールで連絡: liu.11085@osu.edu
