import numpy as np
import pandas as pd
import os

class LSTMDataLogger:
    def __init__(self, sequence_length=60):
        """
        Initializes the Logger for Task 3.
        sequence_length (T) is set to 60 as per Task 3.2 description.
        """
        self.sequence_length = sequence_length
        # This list will hold raw rows of data before converting to Tensors
        self.history = [] 

    def check_resource_starvation(self, bot) -> int:
        """
        TASK 3.3: Resource Starvation Labeling
        Monitors the game state to detect severe resource shortage.
        Returns 1 if starving, 0 otherwise.
        """
        # Example Logic: We have almost no minerals/gas, but our supply is high 
        # (meaning we have an army/workers but are economically choked).
        is_starving = (bot.minerals < 50) and (bot.vespene < 50) and (bot.supply_used > 30)
        
        return 1 if is_starving else 0

    def log_step(self, features_array: np.ndarray, bot):
        """
        Takes the array from Week 1 (features), appends the Starvation Label (Week 3), 
        and adds it to the timeline.
        """
        # Get the label (0 or 1)
        starvation_label = self.check_resource_starvation(bot)
        
        # Combine the features with the label as the last column
        row = np.append(features_array, starvation_label)
        
        # Add to our timeline
        self.history.append(row)

    def save_step_to_csv(self, filename="sc2_game_data.csv"):
        """
        TASK 3.1: Save to CSV and Parquet.
        Saves the linear sequence of game steps to disk.
        """
        if len(self.history) == 0:
            print("[Data Logger] No data to save.")
            return

        # Convert history to a Pandas DataFrame
        df = pd.DataFrame(self.history)
        
        # Create output directory
        os.makedirs("data", exist_ok=True)
        csv_path = os.path.join("data", filename)
        
        # 1. Save as CSV
        df.to_csv(csv_path, index=False, header=False)
        print(f"[Task 3.1] Data saved to CSV: {csv_path}")

        # 2. Save as Parquet (Added Value for extreme speed and small file size)
        parquet_path = csv_path.replace('.csv', '.parquet')
        df.columns = [str(c) for c in df.columns] # Parquet requires string headers
        df.to_parquet(parquet_path, engine='pyarrow')
        print(f"[Task 3.1] Data saved to Parquet: {parquet_path}")

    def generate_lstm_tensors(self) -> tuple:
        """
        TASK 3.2: Time-Windowing for LSTM.
        Converts consecutive rows into chunks of T=60.
        Returns:
            X_tensor: Shape (Batch, Sequence_Length, Features)
            y_tensor: Shape (Batch, 1)
        """
        data = np.array(self.history)
        
        # If we don't have enough data for even one sequence, abort
        if len(data) < self.sequence_length:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length} steps.")

        # X is everything except the last column, y is the last column (Label)
        X_raw = data[:, :-1]
        y_raw = data[:, -1]
        
        X_windows = []
        y_labels = []
        
        # Sliding Window Magic
        for i in range(len(X_raw) - self.sequence_length + 1):
            # Grab a chunk of T=60 rows
            window = X_raw[i : i + self.sequence_length]
            
            # The label for this window is the starvation status at the END of the 60 seconds
            target_label = y_raw[i + self.sequence_length - 1]
            
            X_windows.append(window)
            y_labels.append(target_label)
            
        # Convert lists to NumPy Tensors
        X_tensor = np.array(X_windows, dtype=np.float32)
        y_tensor = np.array(y_labels, dtype=np.int32)
        
        # Formatting print to prove Task 3.2 is fulfilled
        print(f"[Task 3.2] Final Tensor X Shape: {X_tensor.shape} -> (Batch, Sequence_Length, Features)")
        print(f"[Task 3.2] Final Tensor y Shape: {y_tensor.shape} -> (Batch, Label)")
        
        return X_tensor, y_tensor