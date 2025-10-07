# TSB-AD プロジェクトサマリー

## 📋 プロジェクト概要

このプロジェクトは、時系列異常検知のための包括的なベンチマークシステム「TSB-AD」です。
他のサーバーでも簡単に使用できるように、GitHubを使用した管理体制を整備しました。

## ✅ 完了した作業

### 1. ドキュメント整備

以下の5つのドキュメントを作成しました:

| ファイル名 | 目的 | 内容 |
|-----------|------|------|
| `SETUP_GUIDE.md` | セットアップ手順 | 環境構築からデータセットのダウンロードまでの詳細な手順 |
| `QUICKSTART.md` | クイックスタート | 5分で始められる最小限の手順と基本的な使い方 |
| `DEPLOYMENT.md` | デプロイメント | 他のサーバーへの展開方法（Git/rsync/Docker） |
| `PUSH_TO_GITHUB.md` | GitHub連携 | GitHubへのプッシュ方法と認証手順 |
| `PROJECT_SUMMARY.md` | このファイル | プロジェクト全体のサマリー |

### 2. `.gitignore` の整備

以下のファイル/ディレクトリを除外するように設定:

- **データファイル**: `Datasets/TSB-AD-U/*.csv`, `Datasets/TSB-AD-M/*.csv`
- **ログファイル**: `*.log`, `*.out`, `nohup.out`
- **一時ファイル**: `__pycache__/`, `*.egg-info/`, `build/`
- **結果ファイル**: `benchmark_exp/eval/`, `benchmark_exp/tuning_results/`
- **IDE設定**: `.vscode/`, `.idea/`
- **大規模ライブラリ**: `autogluon/`

### 3. 新機能の追加

- **FrequencyBasedAD**: 周波数ベースの異常検知アルゴリズム
- **並列処理スクリプト**: `HP_Tuning_*_parallel.py`, `Run_Detector_*_parallel.py`
- **評価スクリプト**: `evaluate_frequencybased_results.py`, `average_evaluation.py`
- **分析スクリプト**: `AIanalyze_run_results.py`, `AIgenerate_metrics.py`
- **シェルスクリプト**: `setup_datasets.sh`, `run_job.sh`, `run_detector_u.sh`

### 4. Git管理の準備

✅ 3つのコミットを作成:
1. プロジェクトのポータビリティ向上とセットアップガイド
2. デプロイメントとクイックスタートドキュメント
3. GitHubプッシュ手順

## 📦 プロジェクト構成

```
TSB-AD/
├── 📄 ドキュメント
│   ├── README.md              # メインドキュメント
│   ├── SETUP_GUIDE.md         # セットアップガイド
│   ├── QUICKSTART.md          # クイックスタート
│   ├── DEPLOYMENT.md          # デプロイメント手順
│   ├── PUSH_TO_GITHUB.md      # GitHub連携
│   └── PROJECT_SUMMARY.md     # このファイル
│
├── 🔧 設定ファイル
│   ├── .gitignore             # Git除外設定
│   ├── requirements.txt       # Python依存パッケージ
│   ├── setup.py              # インストールスクリプト
│   └── setup_datasets.sh     # データセットセットアップ
│
├── 📚 メインパッケージ
│   └── TSB_AD/
│       ├── models/           # 異常検知アルゴリズム (40+)
│       ├── evaluation/       # 評価メトリクス
│       ├── utils/           # ユーティリティ
│       ├── HP_list.py       # ハイパーパラメータリスト
│       └── model_wrapper.py # モデルラッパー
│
├── 🧪 ベンチマーク実験
│   └── benchmark_exp/
│       ├── HP_Tuning_*.py           # ハイパーパラメータチューニング
│       ├── Run_Detector_*.py        # 検出器実行
│       ├── *_parallel.py            # 並列処理版
│       ├── evaluate_*.py            # 評価スクリプト
│       ├── AI*.py                   # AI分析スクリプト
│       └── *.sh                     # シェルスクリプト
│
└── 📊 データセット (別途ダウンロード)
    └── Datasets/
        ├── TSB-AD-U/        # 単変量 (870系列)
        ├── TSB-AD-M/        # 多変量 (200系列)
        └── File_List/       # データ分割リスト
```

## 🚀 次のステップ

### 1. GitHubにプッシュ

```bash
# 自分のGitHubリポジトリを作成
# https://github.com/new

# リモートを設定
cd /home/25/nakagomi/TSB-AD
git remote add my-repo https://github.com/あなたのユーザー名/TSB-AD.git

# プッシュ
git push my-repo main
```

詳細は `PUSH_TO_GITHUB.md` を参照。

### 2. 他のサーバーにデプロイ

```bash
# 新しいサーバーで
git clone https://github.com/あなたのユーザー名/TSB-AD.git
cd TSB-AD
conda create -n TSB-AD python=3.11
conda activate TSB-AD
pip install -r requirements.txt
pip install -e .
bash setup_datasets.sh
```

詳細は `DEPLOYMENT.md` を参照。

### 3. 実験の実行

```bash
cd benchmark_exp
python HP_Tuning_U.py      # ハイパーパラメータチューニング
python Run_Detector_U.py   # 検出器実行
```

## 📊 利用可能なアルゴリズム

### 統計的手法 (25種類)
- IForest, OCSVM, LOF, KNN, PCA, HBOS, COPOD, MatrixProfile など

### ニューラルネットワーク (10種類)
- AutoEncoder, LSTMAD, CNN, USAD, AnomalyTransformer, TranAD, TimesNet など

### Foundation Models (5種類)
- OFA, Chronos, MOMENT, TimesFM, Lag-Llama

### 新規追加
- **FrequencyBasedAD**: 周波数領域での異常検知

## 💾 データセット情報

- **TSB-AD-U**: 870の単変量時系列 (約2.5GB)
- **TSB-AD-M**: 200の多変量時系列 (約1.8GB)
- **合計**: 1070時系列、40データセット

ダウンロード先:
- https://www.thedatum.org/datasets/TSB-AD-U.zip
- https://www.thedatum.org/datasets/TSB-AD-M.zip

## 🔧 システム要件

- **Python**: 3.8以上 (推奨: 3.11)
- **PyTorch**: 2.3.0
- **メモリ**: 最低8GB (推奨: 16GB以上)
- **ストレージ**: 約10GB (データセット含む)
- **GPU**: オプション (CUDA 12.1対応)

## 📝 重要な注意事項

### Gitで管理しないもの
- ✅ データセットファイル (`.csv`)
- ✅ ログファイル (`.log`, `.out`)
- ✅ 実験結果 (`eval/`, `tuning_results/`)
- ✅ 一時ファイル (`__pycache__/`, `*.egg-info/`)

### 手動で転送が必要なもの
- データセット (4.3GB) → `setup_datasets.sh`で自動ダウンロード可能
- 実験結果 → 必要に応じてrsyncで転送

## 🔄 更新の同期

```bash
# 変更をコミット
git add .
git commit -m "変更内容の説明"

# プッシュ
git push origin main

# 他のサーバーで取得
git pull origin main
```

## 📞 サポート

### ドキュメント
- 基本的な使い方: `QUICKSTART.md`
- 詳細なセットアップ: `SETUP_GUIDE.md`
- デプロイ方法: `DEPLOYMENT.md`
- GitHub連携: `PUSH_TO_GITHUB.md`

### 連絡先
- Qinghua Liu: liu.11085@osu.edu
- John Paparrizos: paparrizos.1@osu.edu
- GitHub Issues: https://github.com/TheDatumOrg/TSB-AD/issues

## 🎯 プロジェクトの目標達成状況

- ✅ プロジェクトをGit管理下に配置
- ✅ 不要なファイルを除外する`.gitignore`を設定
- ✅ 他のサーバーでのセットアップ手順を文書化
- ✅ GitHubへのプッシュ手順を文書化
- ✅ クイックスタートガイドを作成
- ✅ デプロイメント手順を文書化
- ⏳ GitHubへのプッシュ (次のステップ)
- ⏳ 他のサーバーでの動作確認 (プッシュ後)

## 📈 今後の拡張案

1. **Docker化**: Dockerfileを作成して環境を完全に再現可能に
2. **CI/CD**: GitHub Actionsで自動テスト
3. **ドキュメントサイト**: GitHub Pagesで公開
4. **パッケージ化**: PyPIへの公開
5. **ベンチマーク結果**: 自動更新されるリーダーボード

---

**作成日**: 2025年10月7日  
**バージョン**: 1.5  
**ステータス**: GitHub プッシュ準備完了 ✅
