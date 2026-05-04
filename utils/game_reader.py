import numpy as np

class StateExtractor:
    def __init__(self):
        # Variables to store the previous state to calculate collection/spending rates
        self.previous_minerals = 0
        self.previous_time = 0.0

    def get_raw_dict(self, bot) -> dict:
        """
        Returns the raw data as a dictionary. 
        Useful for logging or dashboard displays.
        """
        return {
            "minerals": bot.minerals,
            "vespene": bot.vespene,
            "supply_used": bot.supply_used,
            "supply_cap": bot.supply_cap,
            "workers_count": bot.supply_workers, 
            "idle_workers": bot.workers.idle.amount, 
            "townhalls_count": bot.townhalls.amount 
        }

    def get_tensor_array(self, bot) -> np.ndarray:
        """
        Returns the raw data as a NumPy array.
        Required for Task 1.3 (Scaling) and PyTorch models.
        """
        state_list = [
            bot.minerals,
            bot.vespene,
            bot.supply_used,
            bot.supply_cap,
            bot.supply_workers
        ]
        return np.array(state_list, dtype=np.float32)

    def get_derived_features(self, bot) -> dict:
        """
        Calculates derived features: collection rate, human spending rate, 
        and saturation ratio based on the time and resource differences.
        """
        current_minerals = bot.minerals
        current_time = bot.time # Current in-game time in seconds
        
        # 1 & 2: Calculate Collection Rate and Spending Rate
        time_diff = current_time - self.previous_time
        mineral_diff = current_minerals - self.previous_minerals
        
        collection_rate_per_min = 0.0
        human_spending_rate = 0.0
        
        # Ensure time has passed to avoid division by zero
        if time_diff > 0:
            if mineral_diff > 0:
                # Resources increased = We are collecting
                rate_per_sec = mineral_diff / time_diff
                collection_rate_per_min = rate_per_sec * 60
            elif mineral_diff < 0:
                # Resources decreased = Human player is spending
                # Take the absolute value to represent the positive spending rate
                spending_per_sec = abs(mineral_diff) / time_diff
                human_spending_rate = spending_per_sec * 60

        # 3: Calculate Saturation Ratio
        total_assigned_workers = 0
        total_ideal_workers = 0
        
        # Loop through all completed and ready bases (townhalls)
        for townhall in bot.townhalls.ready:
            # The game API provides these variables ready for each base
            total_assigned_workers += townhall.assigned_harvesters
            total_ideal_workers += townhall.ideal_harvesters
            
        if total_ideal_workers > 0:
            saturation_ratio = total_assigned_workers / total_ideal_workers
        else:
            # If there are no bases at all
            saturation_ratio = 0.0 

        # Update variables for the next step comparison
        self.previous_minerals = current_minerals
        self.previous_time = current_time
        
        return {
            "collection_rate": collection_rate_per_min,
            "spending_rate": human_spending_rate,
            "saturation_ratio": saturation_ratio
        }