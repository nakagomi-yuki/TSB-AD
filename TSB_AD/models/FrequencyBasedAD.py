# -*- coding: utf-8 -*-
# Author: Custom implementation based on FrequencyBasedAD
# License: Apache-2.0 License

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from .base import BaseDetector
from ..utils.slidingWindows import find_length_rank

class FrequencyBasedAD(BaseDetector):
    """
    頻度ベース異常検知モデル
    
    時系列データをスライディングウィンドウで区切り、
    各ウィンドウ内のパターンの出現頻度に基づいて異常を検知する
    """
    
    def __init__(self, periodicity: int = 2, stride_ratio: float = 0.01, 
                 error_tolerance: float = 0.1, min_frequency: float = 0.05,
                 normalize: bool = True, contamination: float = 0.1):
        """
        Parameters:
        -----------
        periodicity : int
            周期性のランク（1, 2, 3）- find_length_rankで使用
        stride_ratio : float
            ストライド幅の比率（ウィンドウサイズに対する割合）
        error_tolerance : float
            測定誤差の許容範囲（正規化後の値）
        min_frequency : float
            異常と判定する最小頻度閾値
        normalize : bool
            データを正規化するかどうか
        contamination : float
            BaseDetectorで必要な汚染率パラメータ
        """
        super().__init__(contamination=contamination)
        self.periodicity = periodicity
        self.stride_ratio = stride_ratio
        self.error_tolerance = error_tolerance
        self.min_frequency = min_frequency
        self.normalize = normalize
        
        # 動的に決定されるパラメータ
        self.window_size = None
        self.stride = None
        
        self.scaler = StandardScaler()
        self.pattern_frequencies = {}
        self.total_windows = 0
        
    def _determine_window_size(self, data: np.ndarray) -> int:
        """
        データの特性に基づいてウィンドウサイズを決定
        """
        # find_length_rankを使用してウィンドウサイズを決定
        window_size = find_length_rank(data, rank=self.periodicity)
        
        # 最小・最大サイズの制限
        min_size = 10
        max_size = min(500, len(data) // 4)
        
        window_size = max(min_size, min(window_size, max_size))
        
        return window_size
    
    def _determine_stride(self, window_size: int) -> int:
        """
        ウィンドウサイズに基づいてストライドを決定
        """
        stride = max(1, int(window_size * self.stride_ratio))
        return stride
    
    def _extract_windows(self, data: np.ndarray) -> List[np.ndarray]:
        """
        時系列データからスライディングウィンドウを抽出
        """
        if self.window_size is None:
            self.window_size = self._determine_window_size(data)
            self.stride = self._determine_stride(self.window_size)
        
        windows = []
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def _quantize_pattern(self, window: np.ndarray) -> Tuple:
        """
        ウィンドウパターンを量子化して誤差許容範囲内で同じパターンとみなす
        """
        # パターンを誤差許容範囲で量子化
        quantized = np.round(window / self.error_tolerance) * self.error_tolerance
        
        # パターンの特徴量を計算
        features = [
            np.mean(quantized),      # 平均値
            np.std(quantized),       # 標準偏差
            np.polyfit(np.arange(len(quantized)), quantized, 1)[0],  # 線形トレンド
            np.max(quantized) - np.min(quantized),  # レンジ
            np.percentile(quantized, 75) - np.percentile(quantized, 25),  # IQR
            np.sum(np.diff(quantized) > 0) / len(quantized),  # 上昇率
            np.sum(np.abs(np.diff(quantized))) / len(quantized)  # 変動強度
        ]
        
        # 特徴量も量子化
        quantized_features = np.round(np.array(features) / self.error_tolerance) * self.error_tolerance
        # quantized_features = np.array(features)  # 量子化しない
        return tuple(quantized_features)
    
    def fit(self, X, y=None):
        """
        モデルの学習（パターンの頻度計算）
        """
        data = X.squeeze() if len(X.shape) > 1 else X
        
        if self.normalize:
            data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # ウィンドウサイズとストライドを決定
        if self.window_size is None:
            self.window_size = self._determine_window_size(data)
            self.stride = self._determine_stride(self.window_size)
        
        # ウィンドウを抽出
        windows = self._extract_windows(data)
        self.total_windows = len(windows)
        
        # 各ウィンドウのパターンを分析
        for window in windows:
            pattern_key = self._quantize_pattern(window)
            
            if pattern_key in self.pattern_frequencies:
                self.pattern_frequencies[pattern_key] += 1
            else:
                self.pattern_frequencies[pattern_key] = 1
        
        # 頻度を正規化
        for key in self.pattern_frequencies:
            self.pattern_frequencies[key] /= self.total_windows
        
        # BaseDetectorの要件に合わせてdecision_scores_を設定
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
            
        return self
    
    def decision_function(self, X):
        """
        異常検知の実行
        """
        data = X.squeeze() if len(X.shape) > 1 else X
        
        if self.normalize:
            data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # ウィンドウを抽出
        windows = self._extract_windows(data)
        anomaly_scores = np.zeros(len(data))
        
        for i, window in enumerate(windows):
            pattern_key = self._quantize_pattern(window)
            
            # パターンの頻度を取得
            frequency = self.pattern_frequencies.get(pattern_key, 0.0)
            
            # 異常スコアを計算（頻度が低いほど異常スコアが高い）
            if frequency < self.min_frequency:
                anomaly_score = 1.0 - frequency
            else:
                anomaly_score = 0.0
            
            # ウィンドウ内の全ポイントに異常スコアを割り当て
            start_idx = i * self.stride
            end_idx = min(start_idx + self.window_size, len(data))
            anomaly_scores[start_idx:end_idx] = np.maximum(
                anomaly_scores[start_idx:end_idx], 
                anomaly_score
            )
        
        return anomaly_scores