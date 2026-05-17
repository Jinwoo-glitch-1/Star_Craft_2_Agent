import torch
import torch.nn as nn

class StarvationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.2):
        """
        TASK 4.1: Building the Model Architecture
        
        Args:
            input_size: Number of features extracted per step (e.g., 8).
            hidden_size: Number of neurons in the hidden layers.
            num_layers: Number of stacked LSTM layers.
            dropout_rate: Prevents overfitting by randomly turning off neurons.
        """
        super(StarvationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. LSTM Layer
        # batch_first=True makes sure the input shape is (Batch, Seq, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 2. Fully Connected Layer
        # Takes the output of the LSTM and squashes it down to 1 output node
        self.fc = nn.Linear(hidden_size, 1)
        
        # 3. Activation Function
        # Sigmoid converts the raw output number into a probability between 0.0 and 1.0
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines how the data flows through the network.
        x shape expected: (batch_size, sequence_length=60, features)
        """
        # Initialize hidden state (h0) and cell state (c0) with zeros
        # These act as the "memory" of the network
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through LSTM
        # 'out' contains the outputs for EVERY time step in the sequence
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # We only care about the prediction at the LAST time step (T=60)
        # So we slice the tensor to get: out[:, -1, :]
        last_time_step_out = out[:, -1, :]
        
        # Pass the last time step's output through the fully connected layer
        dense_out = self.fc(last_time_step_out)
        
        # Apply Sigmoid to get the final probability score
        final_prediction = self.sigmoid(dense_out)
        
        return final_prediction

# ==========================================
# Test the Architecture
# ==========================================
if __name__ == "__main__":
    # Simulate a fake batch of data just to test if the model works
    # Batch=32 games, Sequence=60 steps, Features=8
    fake_data = torch.rand(32, 60, 8) 
    
    # Initialize the model
    model = StarvationLSTM(input_size=8)
    
    # Pass data through the model
    predictions = model(fake_data)
    
    print(f"Input Shape: {fake_data.shape}")
    print(f"Output Shape: {predictions.shape}") 
    print("Architecture built successfully! ✅")