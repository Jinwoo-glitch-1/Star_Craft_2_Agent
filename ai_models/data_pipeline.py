import numpy as np
from utils.game_reader import StateExtractor
from ai_models.feature_scaler import FeatureScaler

def get_final_ai_input(bot, extractor, scaler) -> np.ndarray:
    """
    Combines Tasks 1.1, 1.2, and 1.3 into a single pipeline.
    """
    # 1. Get raw features array (Task 1.1)
    # [minerals, vespene, supply_used, supply_cap, workers_count]
    raw_array = extractor.get_tensor_array(bot)
    
    # 2. Get derived features dict (Task 1.2)
    derived_dict = extractor.get_derived_features(bot)
    derived_array = np.array([
        derived_dict["collection_rate"],
        derived_dict["spending_rate"],
        derived_dict["saturation_ratio"]
    ], dtype=np.float32)
    
    # 3. Concatenate them into one large array
    full_raw_features = np.concatenate((raw_array, derived_array))
    
    # 4. Apply Min-Max Scaling (Task 1.3)
    final_scaled_input = scaler.transform(full_raw_features)
    
    return final_scaled_input