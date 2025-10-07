# 他のサーバーへのデプロイ手順

このドキュメントでは、TSB-ADプロジェクトを他のサーバーにデプロイする手順を説明します。

## オプション1: GitHubを使用したデプロイ (推奨)

### 前提条件
- 新しいサーバーにGitがインストールされていること
- GitHubアカウントを持っていること

### 手順

#### 1. GitHubに自分のリポジトリを作成

1. GitHubにログイン
2. 新しいリポジトリを作成 (例: `your-username/TSB-AD`)
3. リポジトリをプライベートまたはパブリックに設定

#### 2. 現在のサーバーでリモートを変更

```bash
cd /home/25/nakagomi/TSB-AD

# 現在のリモートを確認
git remote -v

# 自分のリポジトリをリモートとして追加 (HTTPSの場合)
git remote add my-origin https://github.com/your-username/TSB-AD.git

# または既存のoriginを変更
git remote set-url origin https://github.com/your-username/TSB-AD.git

# プッシュ
git push -u origin main
```

**SSHを使用する場合:**

```bash
# SSHキーを設定済みの場合
git remote add my-origin git@github.com:your-username/TSB-AD.git
git push -u my-origin main
```

#### 3. 新しいサーバーでクローン

```bash
# 新しいサーバーにログイン
ssh user@new-server

# リポジトリをクローン
git clone https://github.com/your-username/TSB-AD.git
cd TSB-AD

# セットアップを実行
conda create -n TSB-AD python=3.11
conda activate TSB-AD
pip install -r requirements.txt
pip install -e .

# データセットをダウンロード
bash setup_datasets.sh
```

## オプション2: 直接転送 (Gitなし)

### rsyncを使用

```bash
# 現在のサーバーから新しいサーバーへ転送
rsync -avz --exclude='Datasets/TSB-AD-U/*.csv' \
           --exclude='Datasets/TSB-AD-M/*.csv' \
           --exclude='__pycache__' \
           --exclude='*.log' \
           --exclude='*.out' \
           /home/25/nakagomi/TSB-AD/ \
           user@new-server:/path/to/TSB-AD/
```

### scpを使用

```bash
# プロジェクトをtarで圧縮
cd /home/25/nakagomi
tar --exclude='TSB-AD/Datasets/TSB-AD-U/*.csv' \
    --exclude='TSB-AD/Datasets/TSB-AD-M/*.csv' \
    --exclude='TSB-AD/__pycache__' \
    --exclude='TSB-AD/*.log' \
    -czf TSB-AD.tar.gz TSB-AD/

# 新しいサーバーに転送
scp TSB-AD.tar.gz user@new-server:/path/to/destination/

# 新しいサーバーで展開
ssh user@new-server
cd /path/to/destination/
tar -xzf TSB-AD.tar.gz
cd TSB-AD
```

## オプション3: Docker を使用 (将来的な拡張)

Dockerfileを作成して、環境を完全に再現可能にすることもできます。

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["bash"]
```

## セットアップ後の確認

どの方法を使用した場合でも、以下のコマンドでセットアップを確認してください:

```bash
# 環境をアクティベート
conda activate TSB-AD

# 簡単なテストを実行
python -m TSB_AD.main --AD_Name IForest

# または
python -c "from TSB_AD.model_wrapper import run_Unsupervise_AD; print('Setup successful!')"
```

## データセットの同期

データセットは大きいため、Gitリポジトリには含めません。新しいサーバーで以下のいずれかの方法でデータセットを取得してください:

### 方法1: 公式サイトからダウンロード

```bash
cd Datasets
wget https://www.thedatum.org/datasets/TSB-AD-U.zip
wget https://www.thedatum.org/datasets/TSB-AD-M.zip
unzip TSB-AD-U.zip
unzip TSB-AD-M.zip
```

### 方法2: サーバー間で直接転送

```bash
# 元のサーバーから新しいサーバーへ
rsync -avz /home/25/nakagomi/TSB-AD/Datasets/ \
           user@new-server:/path/to/TSB-AD/Datasets/
```

## トラブルシューティング

### 認証エラー

GitHubへのプッシュ時に認証エラーが発生した場合:

1. **Personal Access Token (PAT)を使用:**
   - GitHubの Settings > Developer settings > Personal access tokens
   - 新しいトークンを生成
   - プッシュ時にパスワードの代わりにトークンを使用

2. **SSH キーを使用:**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   cat ~/.ssh/id_ed25519.pub
   # 表示された公開鍵をGitHubのSSH keysに追加
   ```

### 権限エラー

```bash
# ファイルの権限を修正
chmod +x benchmark_exp/*.sh
chmod +x setup_datasets.sh
```

## 更新の同期

変更を加えた後、他のサーバーに同期する方法:

```bash
# 現在のサーバーで
git add .
git commit -m "Your changes description"
git push origin main

# 新しいサーバーで
git pull origin main
```

## 注意事項

1. **大きなファイル**: データセットやログファイルは`.gitignore`で除外されています
2. **機密情報**: APIキーや認証情報をコミットしないでください
3. **ブランチ戦略**: 実験的な変更は別ブランチで行うことを推奨します

```bash
# 新しいブランチを作成
git checkout -b experiment-feature

# 変更をコミット
git add .
git commit -m "Experimental feature"

# プッシュ
git push origin experiment-feature
```
