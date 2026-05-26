import numpy as np
from utils.constants import FEATURES_ORDER

class FeatureScaler:
    def __init__(self):
        # القاموس المركزي للقيم القصوى (9 Features)
        self.max_vals_dict = {
            "Minerals": 3000.0,
            "Gas": 2000.0,
            "Supply_Used": 200.0,
            "Supply_Cap": 200.0,
            "Workers": 80.0,
            "Active_Bases": 10.0, 
            "Collection_Rate": 2000.0,
            "Spending_Rate": 2000.0,
            "Saturation_Ratio": 1.0
        }
        
        self.min_vals = np.zeros(len(FEATURES_ORDER), dtype=np.float32)
        # استخراج القيم القصوى بنفس الترتيب الثابت
        self.max_vals = np.array([self.max_vals_dict[f] for f in FEATURES_ORDER], dtype=np.float32)

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0 
        
        scaled_features = (raw_features - self.min_vals) / range_vals
        scaled_features = np.clip(scaled_features, 0.0, 1.0)
        return scaled_features