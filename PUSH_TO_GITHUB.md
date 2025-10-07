# GitHubへのプッシュ手順

このドキュメントでは、現在のプロジェクトをGitHubにプッシュする手順を説明します。

## 現在の状態

✅ 2つのコミットが準備完了:
1. プロジェクトのポータビリティ向上とセットアップガイドの追加
2. デプロイメントとクイックスタートドキュメントの追加

## オプション1: 自分のGitHubリポジトリにプッシュ (推奨)

### ステップ1: GitHubで新しいリポジトリを作成

1. https://github.com にアクセス
2. 右上の「+」→「New repository」をクリック
3. リポジトリ名を入力 (例: `TSB-AD`)
4. プライベートまたはパブリックを選択
5. **「Initialize this repository with a README」のチェックを外す**
6. 「Create repository」をクリック

### ステップ2: リモートリポジトリを設定

```bash
cd /home/25/nakagomi/TSB-AD

# 現在のリモートを確認
git remote -v

# 自分のリポジトリを新しいリモートとして追加
git remote add my-repo https://github.com/あなたのユーザー名/TSB-AD.git

# または、既存のoriginを変更する場合
git remote set-url origin https://github.com/あなたのユーザー名/TSB-AD.git
```

### ステップ3: プッシュ

```bash
# 自分のリポジトリにプッシュ
git push my-repo main

# または、originを変更した場合
git push origin main
```

### 認証方法

#### 方法A: Personal Access Token (PAT)

1. GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 「Generate new token」をクリック
3. スコープで「repo」を選択
4. トークンを生成してコピー
5. プッシュ時にパスワードの代わりにトークンを使用

```bash
Username: あなたのGitHubユーザー名
Password: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (生成したトークン)
```

#### 方法B: SSH キー

```bash
# SSH キーを生成 (まだない場合)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 公開鍵を表示
cat ~/.ssh/id_ed25519.pub

# 表示された内容をGitHub Settings → SSH and GPG keys に追加

# SSHでリモートを設定
git remote set-url origin git@github.com:あなたのユーザー名/TSB-AD.git

# プッシュ
git push origin main
```

## オプション2: 元のリポジトリにプルリクエスト

元のTSB-ADプロジェクトに貢献したい場合:

### ステップ1: フォークを作成

1. https://github.com/TheDatumOrg/TSB-AD にアクセス
2. 右上の「Fork」ボタンをクリック

### ステップ2: フォークをリモートに追加

```bash
cd /home/25/nakagomi/TSB-AD

# フォークをリモートに追加
git remote add fork https://github.com/あなたのユーザー名/TSB-AD.git

# フォークにプッシュ
git push fork main
```

### ステップ3: プルリクエストを作成

1. あなたのフォークのページにアクセス
2. 「Pull request」をクリック
3. 変更内容を説明
4. 「Create pull request」をクリック

## プッシュ後の確認

```bash
# プッシュが成功したか確認
git log --oneline -5

# リモートの状態を確認
git remote -v

# ブランチの状態を確認
git status
```

## 他のサーバーでクローン

プッシュが成功したら、他のサーバーで以下のコマンドでクローンできます:

```bash
# 新しいサーバーにログイン
ssh user@new-server

# クローン
git clone https://github.com/あなたのユーザー名/TSB-AD.git
cd TSB-AD

# セットアップ
conda create -n TSB-AD python=3.11
conda activate TSB-AD
pip install -r requirements.txt
pip install -e .
bash setup_datasets.sh
```

## トラブルシューティング

### エラー: "failed to push some refs"

```bash
# リモートの最新状態を取得
git fetch origin

# マージまたはリベース
git pull origin main --rebase

# 再度プッシュ
git push origin main
```

### エラー: "Permission denied (publickey)"

SSHキーが正しく設定されていません:

```bash
# SSH接続をテスト
ssh -T git@github.com

# HTTPSに切り替える
git remote set-url origin https://github.com/あなたのユーザー名/TSB-AD.git
```

### エラー: "Authentication failed"

Personal Access Tokenを使用してください (上記の方法A参照)。

## 更新の同期

変更を加えた後:

```bash
# 変更をステージング
git add .

# コミット
git commit -m "変更の説明"

# プッシュ
git push origin main
```

## ブランチ戦略 (推奨)

実験的な変更は別ブランチで:

```bash
# 新しいブランチを作成
git checkout -b feature/new-algorithm

# 変更をコミット
git add .
git commit -m "Add new algorithm"

# プッシュ
git push origin feature/new-algorithm
```

## 注意事項

⚠️ **以下のファイルはプッシュしないでください:**
- データセットファイル (`.csv`)
- ログファイル (`.log`, `.out`)
- 一時ファイル
- 機密情報 (APIキー、パスワード)

これらは`.gitignore`で除外されています。

## 次のステップ

1. ✅ GitHubにプッシュ
2. 📝 README.mdを更新 (必要に応じて)
3. 🏷️ タグを作成 (バージョン管理)
4. 🚀 他のサーバーにデプロイ

詳細は [DEPLOYMENT.md](DEPLOYMENT.md) を参照してください。
