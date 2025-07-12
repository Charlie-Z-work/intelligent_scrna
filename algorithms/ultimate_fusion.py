#!/usr/bin/env python3
"""简化终极融合"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class UltimateFusion:
    def __init__(self, config=None):
        self.config = config
    
    def fit_predict(self, X, strategy):
        n_clusters = strategy.get('n_clusters', 3)
        pca_components = strategy.get('pca_components', 50)
        random_state = strategy.get('random_state', 42)
        
        if pca_components > 0 and pca_components < X.shape[1]:
            pca = PCA(n_components=pca_components, random_state=random_state)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        # 简化版：只用GMM
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', 
                             random_state=random_state)
        return gmm.fit_predict(X_reduced)
