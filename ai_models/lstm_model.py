import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# TASK 4.1: Building the Model Architecture
# ==========================================
class ResourcePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.2):
        """
        Architecture: LSTM -> Fully Connected Layers (No Sigmoid for Regression)
        """
        super(ResourcePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. LSTM Layer to extract temporal features
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 2. Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        # Output layer for continuous numerical value (Regression)
        self.fc2 = nn.Linear(32, 1) 

    def forward(self, x):
        out, _ = self.lstm(x)
        last_time_step_out = out[:, -1, :]
        
        dense1 = self.fc1(last_time_step_out)
        activated = self.relu(dense1)
        final_prediction = self.fc2(activated) 
        
        return final_prediction

# ==========================================
# TASK 4.2: Defining the cost function and optimizer
# ==========================================
def setup_training(model, learning_rate=0.001):
    """
    Using MSELoss for numerical prediction and AdamW optimizer.
    """
    # Cost Function: Mean Squared Error
    criterion = nn.MSELoss()
    
    # Optimizer: AdamW
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    return criterion, optimizer

# ==========================================
# Test the Setup
# ==========================================
if __name__ == "__main__":
    model = ResourcePredictor(input_size=8)
    loss_fn, optimizer = setup_training(model)
    
    print("Tasks 4.1 & 4.2 Built Successfully! ✅")
    print(f"Model Name: {model.__class__.__name__}")
    print(f"Loss Function: {loss_fn.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")