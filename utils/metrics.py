#!/usr/bin/env python3
"""简化指标计算器"""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

class MetricsCalculator:
    def calculate_all_metrics(self, y_true, y_pred, X):
        metrics = {}
        try:
            metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
            metrics['ari'] = adjusted_rand_score(y_true, y_pred)
            metrics['silhouette'] = silhouette_score(X, y_pred)
        except:
            metrics = {'nmi': 0.0, 'ari': 0.0, 'silhouette': 0.0}
        return metrics
