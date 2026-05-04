import numpy as np

class FeatureScaler:
    def __init__(self):
        """
        Initialize the Min and Max values for each feature based on StarCraft 2 mechanics.
        Feature Order: 
        [minerals, vespene, supply_used, supply_cap, workers_count, collection_rate, spending_rate, saturation_ratio]
        """
        # Minimum possible values (mostly zeros)
        self.min_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Maximum expected values in a normal macro game
        # You can adjust these later during Hyperparameter Tuning (Task 6.2)
        self.max_vals = np.array([
            3000.0,  # Max Minerals (capped at 3000 for scaling purposes)
            2000.0,  # Max Vespene Gas
            200.0,   # Max Supply Used
            200.0,   # Max Supply Cap
            80.0,    # Max Workers Count
            2000.0,  # Max Collection Rate per minute
            2000.0,  # Max Spending Rate per minute
            1.0      # Saturation Ratio is already between 0 and 1 (100%)
        ], dtype=np.float32)

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Applies Min-Max scaling to manually normalize the array to the [0, 1] range.
        Formula: (X - X_min) / (X_max - X_min)
        """
        # Calculate the range (denominator)
        range_vals = self.max_vals - self.min_vals
        
        # Avoid division by zero (just as a safety measure)
        range_vals[range_vals == 0] = 1.0 
        
        # Apply the scaling formula
        scaled_features = (raw_features - self.min_vals) / range_vals
        
        # Clip the values to strictly stay within [0, 1]
        # This handles cases where in-game resources exceed our expected max_vals (e.g., having 4000 minerals)
        scaled_features = np.clip(scaled_features, 0.0, 1.0)
        
        return scaled_features