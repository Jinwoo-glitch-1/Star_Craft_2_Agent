import numpy as np

def get_final_ai_input(bot, extractor, scaler) -> np.ndarray:
    """يسحب البيانات كـ Vector مرتب، ويطبق عليها الـ Scaling فوراً"""
    raw_vector = extractor.get_feature_vector(bot)
    final_scaled_input = scaler.transform(raw_vector)
    return final_scaled_input