#!/bin/bash

# TSB-AD データセットセットアップスクリプト

echo "TSB-AD データセットのセットアップを開始します..."

# 必要なディレクトリを作成
mkdir -p Datasets
cd Datasets

# データセットのダウンロード
echo "単変量時系列データセット（TSB-AD-U）をダウンロード中..."
if command -v wget &> /dev/null; then
    wget https://www.thedatum.org/datasets/TSB-AD-U.zip -O TSB-AD-U.zip
elif command -v curl &> /dev/null; then
    curl -L https://www.thedatum.org/datasets/TSB-AD-U.zip -o TSB-AD-U.zip
else
    echo "エラー: wget または curl が見つかりません。"
    exit 1
fi

echo "多変量時系列データセット（TSB-AD-M）をダウンロード中..."
if command -v wget &> /dev/null; then
    wget https://www.thedatum.org/datasets/TSB-AD-M.zip -O TSB-AD-M.zip
elif command -v curl &> /dev/null; then
    curl -L https://www.thedatum.org/datasets/TSB-AD-M.zip -o TSB-AD-M.zip
fi

# ファイルの存在確認
if [ ! -f "TSB-AD-U.zip" ]; then
    echo "エラー: TSB-AD-U.zip のダウンロードに失敗しました。"
    exit 1
fi

if [ ! -f "TSB-AD-M.zip" ]; then
    echo "エラー: TSB-AD-M.zip のダウンロードに失敗しました。"
    exit 1
fi

# ZIPファイルの展開
echo "ファイルを展開中..."
if command -v unzip &> /dev/null; then
    unzip -o TSB-AD-U.zip
    unzip -o TSB-AD-M.zip
else
    echo "エラー: unzip が見つかりません。"
    echo "以下のコマンドでインストールしてください："
    echo "  Ubuntu/Debian: sudo apt-get install unzip"
    echo "  CentOS/RHEL: sudo yum install unzip"
    exit 1
fi

# 展開後の確認
echo "展開後のディレクトリ構造を確認中..."
if [ -d "TSB-AD-U" ] && [ -d "TSB-AD-M" ]; then
    echo "✅ データセットのセットアップが完了しました！"
    echo ""
    echo "ディレクトリ構造:"
    ls -la
    echo ""
    echo "TSB-AD-U ファイル数: $(ls TSB-AD-U/*.csv 2>/dev/null | wc -l)"
    echo "TSB-AD-M ファイル数: $(ls TSB-AD-M/*.csv 2>/dev/null | wc -l)"
    echo ""
    echo "次のステップ:"
    echo "1. cd .."
    echo "2. pip install -r requirements.txt"
    echo "3. pip install -e ."
    echo "4. cd benchmark_exp"
    echo "5. python Run_Detector_U.py --AD_Name IForest --save True"
else
    echo "❌ データセットの展開に失敗しました。"
    exit 1
fi 