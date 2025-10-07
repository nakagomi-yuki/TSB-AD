#!/bin/bash
#PBS -N FrequencyBasedAD_Sequential
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -j oe

# 環境設定
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TSB-AD

# 作業ディレクトリに移動
cd $PBS_O_WORKDIR

# 環境変数の設定
export AD_Name=FrequencyBasedAD
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 実行情報の表示
echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Algorithm: $AD_Name"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "======================================"

# Run_Detector_U.pyの実行（逐次処理）
python Run_Detector_U.py \
    --dataset_dir ../Datasets/TSB-AD-U/ \
    --file_lsit ../Datasets/File_List/TSB-AD-U-Eva.csv \
    --score_dir eval/score/uni/ \
    --save_dir eval/metrics/uni/ \
    --save True \
    --AD_Name $AD_Name

# 終了情報の表示
echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"


