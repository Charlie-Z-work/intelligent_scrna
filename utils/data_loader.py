#!/usr/bin/env python3
"""简化数据加载器"""
import numpy as np
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, config=None):
        self.config = config
    
    def load_dataset(self, data_path, labels_path=None):
        X = pd.read_csv(data_path, header=None).values
        y_true = None
        if labels_path and Path(labels_path).exists():
            y_true = pd.read_csv(labels_path, header=None).values.squeeze()
        return X, y_true
    
    def load_datasets_config(self, config_path):
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
