import numpy as np
import pandas as pd
from sc2.ids.unit_typeid import UnitTypeId
from utils.constants import FEATURES_ORDER

class DataExtractor:
    def __init__(self):
        self.prev_workers = 12
        self.previous_minerals = 0
        self.previous_time = 0.0

    def get_raw_state(self, bot):
        workers = bot.workers.amount
        minerals = bot.minerals
        gas = bot.vespene
        supply_used = bot.supply_used
        supply_cap = bot.supply_cap
        active_bases = bot.townhalls.amount
        
        current_time = bot.time
        time_diff = current_time - self.previous_time
        mineral_diff = minerals - self.previous_minerals
        
        collection_rate = 0.0
        spending_rate = 0.0
        if time_diff > 0:
            if mineral_diff > 0:
                collection_rate = (mineral_diff / time_diff) * 60
            elif mineral_diff < 0:
                spending_rate = (abs(mineral_diff) / time_diff) * 60
        
        total_assigned = sum(th.assigned_harvesters for th in bot.townhalls.ready)
        total_ideal = sum(th.ideal_harvesters for th in bot.townhalls.ready)
        saturation = (total_assigned / total_ideal) if total_ideal > 0 else 0.0

        self.previous_minerals = minerals
        self.previous_time = current_time

        return {
            "Minerals": minerals,
            "Gas": gas,
            "Supply_Used": supply_used,
            "Supply_Cap": supply_cap,
            "Workers": workers,
            "Active_Bases": active_bases,
            "Collection_Rate": collection_rate,
            "Spending_Rate": spending_rate,
            "Saturation_Ratio": saturation
        }

    def get_feature_vector(self, bot) -> np.ndarray:
        state = self.get_raw_state(bot) 
        # بناء الـ Array بناءً على الـ Schema الموحدة (9 أبعاد)
        return np.array([state[f] for f in FEATURES_ORDER], dtype=np.float32)