import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

# استدعاء الموديل
from ai_models.lstm_model import ResourcePredictor, setup_training

# ==========================================
# 1. Dataset Class (تجهيز البيانات)
# ==========================================
class SC2Dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Data Loader Module (سحب الداتا الحقيقية)
# ==========================================
def load_and_prepare_data(csv_path, seq_length=60):
    """
    بيقرأ ملف الـ CSV، وبيطبق نظام الـ Sliding Window عشان يجهز الداتا للـ LSTM
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"⚠️ Data file not found: {csv_path}\nRun main.py first to generate data.")
        
    print(f"📂 Loading real data from {csv_path}...")
    
    # قراءة الداتا (بدون header لأننا لم نقم بحفظ العناوين)
    df = pd.read_csv(csv_path, header=None)
    data = df.values
    
    if len(data) < seq_length:
        raise ValueError(f"⚠️ Not enough data! Need at least {seq_length} rows, found {len(data)}.")

    # فصل الـ Features عن الـ Label
    X_raw = data[:, :-1]  # كل العواميد ما عدا الأخير (الـ 9 ميزات)
    y_raw = data[:, -1]   # العمود الأخير (Starvation Label)
    
    X_windows = []
    y_labels = []
    
    # تطبيق الـ Sliding Window (كل 60 خطوة مع بعض)
    print("🔄 Generating Time-Series Sequences...")
    for i in range(len(X_raw) - seq_length + 1):
        X_windows.append(X_raw[i : i + seq_length])
        # الليبل الخاص بالـ 60 ثانية دي هو حالة البوت في آخر خطوة فيهم
        y_labels.append(y_raw[i + seq_length - 1])
        
    X_tensor = np.array(X_windows, dtype=np.float32)
    y_tensor = np.array(y_labels, dtype=np.float32)
    
    print(f"✅ Data Prepared! X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    return X_tensor, y_tensor

# ==========================================
# 3. Training Loop (حلقة التدريب)
# ==========================================
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    print(f"\n--- Starting Training for {num_epochs} Epochs ---")
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss (MSE): {avg_loss:.4f}")
        
    print("--- Training Completed Successfully! ---")
    return model

# ==========================================
# 4. Save Model (حفظ الموديل)
# ==========================================
def save_model(model, filename="resource_predictor.pth"):
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)
    torch.save(model.state_dict(), filepath)
    print(f"💾 Model saved to {filepath} ✅")

# ==========================================
# Entry Point (نقطة التشغيل الحقيقية)
# ==========================================
if __name__ == "__main__":
    # مسار ملف الداتا اللي main.py بيحفظ فيه
    csv_filepath = os.path.join("data", "training_data.csv")
    
    try:
        # 1. سحب الداتا الحقيقية وتجهيز الـ Windows
        real_X, real_y = load_and_prepare_data(csv_filepath, seq_length=60)
        
        # 2. تجهيز الـ DataLoader
        dataset = SC2Dataset(real_X, real_y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 3. استدعاء الموديل (استنتاج عدد الفيتشرز أوتوماتيكياً من الداتا لتفادي الأخطاء)
        num_features = real_X.shape[2] 
        model = ResourcePredictor(input_size=num_features)
        print(f"🤖 Model Initialized with input_size = {num_features}")
        
        # 4. تجهيز الـ Loss والـ Optimizer
        criterion, optimizer = setup_training(model, learning_rate=0.001)
        
        # 5. التدريب الفعلي
        trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=20)
        
        # 6. حفظ الموديل لاستخدامه جوه اللعبة
        save_model(trained_model)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Hint: Ensure you have run 'main.py' at least once to generate 'data/training_data.csv'.")