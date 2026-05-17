import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ExpansionFuzzyController:
    def __init__(self):
        """
        Initializes the Fuzzy Logic Controller for expansion decisions.
        Strictly follows the descriptions for Task 2.1 and Task 2.2.
        """
        
        # ==========================================
        # TASK 2.1: Linguistic Variables
        # ==========================================
        
        # 1. Inputs (Antecedents)
        # Mineral_Surplus: Range from 0 to 3000
        self.mineral_surplus = ctrl.Antecedent(np.arange(0, 3001, 1), 'mineral_surplus')
        
        # Worker_Saturation: Range from 0.0 to 1.0 (0% to 100%)
        self.worker_saturation = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'worker_saturation')

        # Game_Time: Range from 0 to 3600 seconds (Early, Mid, Late)
        self.game_time = ctrl.Antecedent(np.arange(0, 3601, 1), 'game_time')
        
        # 2. Output (Consequent)
        # Expansion_Desirability: Score from 0 to 100
        self.expansion_desirability = ctrl.Consequent(np.arange(0, 101, 1), 'expansion_desirability')


        # ==========================================
        # TASK 2.2: Membership Functions
        # ==========================================
        # Using Triangular Membership Functions (trimf) [start, peak, end]
        
        # 1. Worker_Saturation (low, medium, high)
        self.worker_saturation['low'] = fuzz.trimf(self.worker_saturation.universe, [0, 0.0, 0.5])
        self.worker_saturation['medium'] = fuzz.trimf(self.worker_saturation.universe, [0.3, 0.55, 0.85])
        # As per description: 'high' saturation starts from 80% (0.8) and reaches peak at 100% (1.0)
        self.worker_saturation['high'] = fuzz.trimf(self.worker_saturation.universe, [0.8, 1.0, 1.0])

        # 2. Mineral_Surplus (low, medium, high)
        self.mineral_surplus['low'] = fuzz.trimf(self.mineral_surplus.universe, [0, 0, 1000])
        self.mineral_surplus['medium'] = fuzz.trimf(self.mineral_surplus.universe, [500, 1250, 2000])
        self.mineral_surplus['high'] = fuzz.trimf(self.mineral_surplus.universe, [1500, 3000, 3000])

        # 3. Game_Time (early, mid, late)
        self.game_time['early'] = fuzz.trimf(self.game_time.universe, [0, 0, 600])
        self.game_time['mid'] = fuzz.trimf(self.game_time.universe, [300, 900, 1500])
        self.game_time['late'] = fuzz.trimf(self.game_time.universe, [1200, 3600, 3600])

        # 4. Expansion_Desirability (reject, delay, execute immediately)
        self.expansion_desirability['reject'] = fuzz.trimf(self.expansion_desirability.universe, [0, 0, 40])
        self.expansion_desirability['delay'] = fuzz.trimf(self.expansion_desirability.universe, [30, 50, 70])
        self.expansion_desirability['execute_immediately'] = fuzz.trimf(self.expansion_desirability.universe, [60, 100, 100])

        # Internal placeholder for the control system
        self.ctrl_system = None
        self.simulator = None

        # ==========================================
        # TASK 2.3: Rule Base
        # ==========================================
        # Description requirement: Writing rules based on experience.
        # Mandatory Example: IF Worker_Saturation is High AND Mineral_Surplus is High 
        # THEN Expansion_Desirability is Execute_Immediately.
        
        # Rule 1: The exact mandatory rule requested in the task description.
        rule1 = ctrl.Rule(self.worker_saturation['high'] & self.mineral_surplus['high'], 
                          self.expansion_desirability['execute_immediately'])
        
        # Rule 2 (Experience): If we are floating minerals (High) but saturation is only Medium, 
        # we should still expand to spend the bank and avoid inefficiency.
        rule2 = ctrl.Rule(self.mineral_surplus['high'] & self.worker_saturation['medium'], 
                          self.expansion_desirability['execute_immediately'])
        
        # Rule 3 (Experience): If we have low minerals, we physically cannot afford a Nexus/Command Center.
        # Therefore, reject expansion regardless of other factors.
        rule3 = ctrl.Rule(self.mineral_surplus['low'], 
                          self.expansion_desirability['reject'])
        
        # Rule 4 (Experience): If worker saturation is low, it means we recently expanded or lost workers. 
        # We need to fill the current base before building a new one. Reject expansion.
        rule4 = ctrl.Rule(self.worker_saturation['low'], 
                          self.expansion_desirability['reject'])
        
        # Rule 5 (Experience): If the base is fully saturated but we only have medium minerals, 
        # delay the expansion until we have a safer bank (avoiding spending all money on a Nexus during an attack).
        rule5 = ctrl.Rule(self.worker_saturation['high'] & self.mineral_surplus['medium'], 
                          self.expansion_desirability['delay'])
        
        # Rule 6 (Experience): In the late game, map control is critical. 
        # If we have high saturation late in the game, secure the next base immediately.
        rule6 = ctrl.Rule(self.game_time['late'] & self.worker_saturation['high'], 
                          self.expansion_desirability['execute_immediately'])

        # ==========================================
        # Building the Control System
        # ==========================================
        # Aggregating all rules and feeding them into the control system
        self.ctrl_system = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6
        ])

    def compute_expansion_score(self, minerals, saturation, time) -> float:
        """
        TASK 2.4: Defuzzification
        This function takes live game data and computes a crisp decision score (0-100).
        Uses the Centroid method to calculate the result.
        """
        # 1. Initialize the Simulator if it's the first time
        if self.simulator is None:
            self.simulator = ctrl.ControlSystemSimulation(self.ctrl_system)

        # 2. Feed the raw inputs into the fuzzy system
        self.simulator.input['mineral_surplus'] = minerals
        self.simulator.input['worker_saturation'] = saturation
        self.simulator.input['game_time'] = time

        # 3. Perform the fuzzy computation
        self.simulator.compute()

        # 4. Extract the crisp output (The Defuzzification step)
        # This gives us a score from 0 to 100
        output_score = self.simulator.output['expansion_desirability']
        
        return output_score

    def should_expand(self, output_score, threshold=75) -> bool:
        """
        Implements the final decision logic as per the description:
        'if output_score > 75: bot.expand_now()'
        """
        return output_score > threshold